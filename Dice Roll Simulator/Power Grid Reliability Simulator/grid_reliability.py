"""Power Grid Reliability Simulator — Cascading Failure Risk Assessment
A Monte Carlo simulation showing how small perturbations cascade into
blackouts, and how renewable intermittency affects grid stability.

Based on research from ETH Zürich cascading failure models, NERC 2026
reliability assessments, and recent 2025 blackout events (Iraq, Iberia).

Key concepts:
  • N-k Contingency: What happens when k components fail simultaneously?
  • Cascading Failure: One line overload → redistribution → more overloads
  • Renewable Intermittency: Solar/wind volatility adds uncertainty
  • Storage Mitigation: Batteries buffer intermittency but have limits
  
The paradox: Adding capacity (renewables) can increase blackout risk
through complex network dynamics and reduced safety margins.

Tags: intermediate, engineering, energy, infrastructure, risk, cli
"""

import argparse
import csv
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import math

# ──────────────────────────────────────────────
#   Power Grid Data Structures
# ──────────────────────────────────────────────

@dataclass
class Bus:
    """A node in the power grid where generation/load connects."""
    id: str
    generation_capacity: float    # MW (max output)
    actual_generation: float      # MW (current output)
    load: float                   # MW (demand at this bus)
    voltage_magnitude: float = 1.0  # p.u. (per unit, nominal = 1.0)
    disconnected: bool = False    # Blackout at this bus?

@dataclass
class TransmissionLine:
    """A connection between two buses."""
    id: str
    from_bus: str
    to_bus: str
    capacity_mva: float       # Maximum power flow (MVA)
    actual_flow: float = 0.0   # Current power flow (MW)
    impedance: float = 0.1     # Ohmic resistance (for power flow calc)
    tripped: bool = False      # Failed/outage?
    overload_factor: float = 1.0  # actual_flow / capacity

@dataclass
class Generator:
    """Power plant connected to the grid."""
    id: str
    bus_id: str
    max_output: float       # MW
    min_output: float = 0.0
    current_output: float = 0.0
    type: str = "thermal"   # thermal, nuclear, hydro, solar, wind
    availability: float = 1.0  # 1.0 = online, 0.0 = offline
    ramp_rate: float = 0.1  # Fraction of capacity that can change per timestep

@dataclass  
class Storage:
    """Battery storage system."""
    id: str
    bus_id: str
    capacity_mwh: float     # Energy storage capacity
    power_rating_mw: float  # Charge/discharge rate
    soc: float = 0.5        # State of charge (0-1)
    efficiency: float = 0.95  # Round-trip efficiency

# ──────────────────────────────────────────────
#   Network Topologies
# ──────────────────────────────────────────────

def build_simple_grid() -> Tuple[List[Bus], List[TransmissionLine], List[Generator], List[Storage]]:
    """
    Simple 5-bus test system (classic IEEE test case approximation):
    
    G1 --(T1)-- B1 --(T2)-- B2 --(T3)-- B3 --(T4)-- B4 --(T5)-- B5
                                     |                   |
                                    L1                  L2
    (G1 = generator, B = bus, T = transmission line, L = load)
    """
    buses = [
        Bus("B1", generation_capacity=500, actual_generation=0, load=100),
        Bus("B2", generation_capacity=300, actual_generation=0, load=150),
        Bus("B3", generation_capacity=400, actual_generation=0, load=200),
        Bus("B4", generation_capacity=200, actual_generation=0, load=100),
        Bus("B5", generation_capacity=100, actual_generation=0, load=50),
    ]
    
    lines = [
        TransmissionLine("T1", "G1", "B1", capacity_mva=400),
        TransmissionLine("T2", "B1", "B2", capacity_mva=350),
        TransmissionLine("T3", "B2", "B3", capacity_mva=400),
        TransmissionLine("T4", "B3", "B4", capacity_mva=250),
        TransmissionLine("T5", "B4", "B5", capacity_mva=200),
        # Parallel paths (redundancy)
        TransmissionLine("T6", "B1", "B3", capacity_mva=300),
        TransmissionLine("T7", "B2", "B4", capacity_mva=200),
    ]
    
    generators = [
        Generator("G1", "B1", max_output=500, type="thermal"),
        Generator("G2", "B2", max_output=300, type="thermal"),
        Generator("G3", "B3", max_output=400, type="nuclear"),
        Generator("G4", "B4", max_output=200, type="hydro"),
    ]
    
    storage = [
        Storage("S1", "B2", capacity_mwh=200, power_rating_mw=100),
    ]
    
    return buses, lines, generators, storage

def build_renewable_heavy_grid() -> Tuple[List[Bus], List[TransmissionLine], List[Generator], List[Storage]]:
    """
    Similar grid but with high renewable penetration (adds intermittency risk).
    """
    buses = [
        Bus("B1", generation_capacity=300, actual_generation=0, load=100),
        Bus("B2", generation_capacity=250, actual_generation=0, load=150),
        Bus("B3", generation_capacity=200, actual_generation=0, load=200),
        Bus("B4", generation_capacity=150, actual_generation=0, load=100),
        Bus("B5", generation_capacity=100, actual_generation=0, load=50),
    ]
    
    lines = [
        TransmissionLine("T1", "G1", "B1", capacity_mva=350),
        TransmissionLine("T2", "B1", "B2", capacity_mva=300),
        TransmissionLine("T3", "B2", "B3", capacity_mva=350),
        TransmissionLine("T4", "B3", "B4", capacity_mva=200),
        TransmissionLine("T5", "B4", "B5", capacity_mva=150),
        TransmissionLine("T6", "B1", "B3", capacity_mva=250),
        TransmissionLine("T7", "B2", "B4", capacity_mva=180),
    ]
    
    generators = [
        Generator("G1", "B1", max_output=300, type="solar"),   # Intermittent
        Generator("G2", "B2", max_output=200, type="wind"),    # Intermittent
        Generator("G3", "B3", max_output=150, type="solar"),   # Intermittent
        Generator("G4", "B4", max_output=150, type="hydro"),
        Generator("G5", "B4", max_output=100, type="thermal"), # Backup
    ]
    
    storage = [
        Storage("S1", "B2", capacity_mwh=300, power_rating_mw=150),
        Storage("S2", "B4", capacity_mwh=200, power_rating_mw=100),
    ]
    
    return buses, lines, generators, storage

def build_stressed_grid() -> Tuple[List[Bus], List[TransmissionLine], List[Generator], List[Storage]]:
    """
    High-demand scenario approaching N-1 contingency limits.
    (Like Texas ERCOT summer 2026 projections)
    """
    buses = [
        Bus("B1", generation_capacity=800, actual_generation=0, load=250),
        Bus("B2", generation_capacity=600, actual_generation=0, load=300),
        Bus("B3", generation_capacity=700, actual_generation=0, load=400),
        Bus("B4", generation_capacity=500, actual_generation=0, load=200),
        Bus("B5", generation_capacity=300, actual_generation=0, load=150),
    ]
    
    lines = [
        TransmissionLine("T1", "G1", "B1", capacity_mva=700),
        TransmissionLine("T2", "B1", "B2", capacity_mva=600),
        TransmissionLine("T3", "B2", "B3", capacity_mva=650),
        TransmissionLine("T4", "B3", "B4", capacity_mva=400),
        TransmissionLine("T5", "B4", "B5", capacity_mva=350),
        # Reduced redundancy (more vulnerable)
        TransmissionLine("T6", "B1", "B3", capacity_mva=200),
    ]
    
    generators = [
        Generator("G1", "B1", max_output=800, type="thermal"),
        Generator("G2", "B2", max_output=600, type="gas_peaker"),
        Generator("G3", "B3", max_output=700, type="coal"),
        Generator("G4", "B4", max_output=500, type="thermal"),
    ]
    
    storage = []  # No storage = higher risk
    
    return buses, lines, generators, storage

# ──────────────────────────────────────────────
#   Power Flow & Load Balancing
# ──────────────────────────────────────────────

def compute_dc_power_flow(
    buses: List[Bus],
    lines: List[TransmissionLine],
    generators: List[Generator],
    demand_factor: float = 1.0,
    renewable_factors: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Simplified DC power flow calculation.
    
    In reality this requires solving nonlinear equations (AC power flow).
    For simulation purposes, we use linear approximations:
    - Total demand must equal total generation (+ losses, ignored here)
    - Power distributes proportionally to line impedances
    - Each line carries its share based on topology
    """
    # Reset all lines
    for line in lines:
        line.actual_flow = 0.0
        line.overload_factor = 0.0
        line.tripped = False
    
    # Apply renewable availability factors
    if renewable_factors:
        for gen in generators:
            if gen.type in ["solar", "wind"]:
                gen.availability = renewable_factors.get(gen.id, 0.5)
    
    # Calculate total demand
    total_demand = sum(b.load * demand_factor for b in buses if not b.disconnected)
    
    # Dispatch generators (simple priority: nuclear/hydro first, then thermal, then renewables)
    dispatch_priority = {"nuclear": 1, "hydro": 2, "thermal": 3, "gas_peaker": 4, "solar": 5, "wind": 5}
    sorted_generators = sorted(generators, key=lambda g: (dispatch_priority.get(g.type, 99), -g.max_output))
    
    remaining_demand = total_demand
    for gen in sorted_generators:
        available_capacity = gen.max_output * gen.availability * gen.ramp_rate
        if gen.type in ["solar", "wind"]:
            # Renewables take what's available
            gen.current_output = min(available_capacity, remaining_demand)
        else:
            # Thermal fills in the rest
            gen.current_output = min(available_capacity, remaining_demand)
        
        remaining_demand -= gen.current_output
    
    # If demand exceeds supply, implement load shedding (blackouts)
    if remaining_demand > 0:
        shed_ratio = remaining_demand / total_demand if total_demand > 0 else 0
        # Shed load from highest-numbered buses first (simple approximation)
        for bus in sorted(buses, key=lambda b: int(b.id.replace("B", "")), reverse=True):
            if remaining_demand <= 0:
                break
            if not bus.disconnected and bus.load > 0:
                shed_amount = min(bus.load * shed_ratio, remaining_demand)
                bus.load -= shed_amount
                remaining_demand -= shed_amount
                if bus.load == 0:
                    bus.disconnected = True
    
    # Distribute generation across transmission network
    # This is a simplification — real power flows follow Kirchhoff's laws
    total_generation = sum(g.current_output for g in generators)
    
    # Assign generation to buses
    bus_generation = defaultdict(float)
    for gen in generators:
        bus_generation[gen.bus_id] += gen.current_output
    
    # Approximate power flows (proportional to load at each bus)
    if total_demand > 0:
        # Simplified: assume power flows from generation centers to load centers
        for line in lines:
            # Estimate flow based on connectivity and relative loads
            from_load = next((b.load for b in buses if b.id == line.from_bus), 0)
            to_load = next((b.load for b in buses if b.id == line.to_bus), 0)
            
            if from_load + to_load > 0:
                # Rough estimate: flow proportional to downstream load
                downstream_load = to_load / (from_load + to_load) if (from_load + to_load) > 0 else 0.5
                total_flow_needed = total_demand
                line.actual_flow = total_flow_needed * downstream_load * random.uniform(0.8, 1.2)
            
            # Check for overload
            if line.capacity_mva > 0:
                line.overload_factor = line.actual_flow / line.capacity_mva
    
    return {
        "total_demand": total_demand,
        "total_generation": total_generation,
        "generation_deficit": max(0, total_demand - total_generation),
        "lines_overloaded": [l.id for l in lines if l.overload_factor > 1.0],
    }

# ──────────────────────────────────────────────
#   Cascading Failure Simulation
# ──────────────────────────────────────────────

def simulate_cascading_failure(
    buses: List[Bus],
    lines: List[TransmissionLine],
    generators: List[Generator],
    storage: List[Storage],
    initial_contingency: Optional[List[str]] = None,
    num_timesteps: int = 10,
    renewable_volatility: float = 0.0,
    use_storage: bool = True,
    progress: bool = True,
) -> Dict:
    """
    Simulate a cascading failure event.
    
    Process:
    1. Initial contingency (line/generator failure or renewable dip)
    2. Recalculate power flow
    3. Identify overloaded lines
    4. Protection systems trip overloaded lines
    5. Power redistributes → more overloads possible
    6. Repeat until stable or blackout
    
    Returns detailed history of each timestep.
    """
    history = {
        "timestep": [],
        "total_load_served": [],
        "lines_tripped": [],
        "buses_blacked_out": [],
        "generation_available": [],
        "storage_discharged": [],
    }
    
    # Deep copy to avoid mutating originals
    b_copy = [Bus(b.id, b.generation_capacity, b.actual_generation, b.load, 
                  b.voltage_magnitude, b.disconnected) for b in buses]
    l_copy = [TransmissionLine(l.id, l.from_bus, l.to_bus, l.capacity_mva,
                               l.actual_flow, l.impedance, l.tripped) for l in lines]
    g_copy = [Generator(g.id, g.bus_id, g.max_output, g.min_output, g.current_output,
                        g.type, g.availability, g.ramp_rate) for g in generators]
    s_copy = [Storage(st.id, st.bus_id, st.capacity_mwh, st.power_rating_mw,
                      st.soc, st.efficiency) for st in storage]
    
    # Apply initial contingency
    if initial_contingency:
        for cid in initial_contingency:
            if cid.startswith("T"):  # Line
                line = next((l for l in l_copy if l.id == cid), None)
                if line:
                    line.tripped = True
            elif cid.startswith("G"):  # Generator
                gen = next((g for g in g_copy if g.id == cid), None)
                if gen:
                    gen.availability = 0.0
            elif cid.startswith("B"):  # Bus
                bus = next((b for b in b_copy if b.id == cid), None)
                if bus:
                    bus.disconnected = True
    
    if progress:
        print(f"\nInitial state: {len([l for l in l_copy if not l.tripped])} lines online")
        print(f"Contingency applied: {initial_contingency}")
        print()
    
    total_initial_load = sum(b.load for b in b_copy)
    
    for t in range(num_timesteps):
        # Apply renewable volatility (if applicable)
        if renewable_volatility > 0:
            for gen in g_copy:
                if gen.type in ["solar", "wind"]:
                    noise = random.gauss(0, renewable_volatility)
                    gen.availability = max(0.1, min(1.0, gen.availability + noise))
        
        # Use storage if available and grid stressed
        storage_discharge = 0.0
        if use_storage and s_copy:
            power_flow_result = compute_dc_power_flow(b_copy, l_copy, g_copy)
            if power_flow_result["generation_deficit"] > 0:
                # Discharge batteries to fill gap
                for st in s_copy:
                    discharge = min(
                        st.soc * st.capacity_mwh,
                        st.power_rating_mw,
                        power_flow_result["generation_deficit"] - storage_discharge
                    )
                    st.soc -= discharge / st.capacity_mwh
                    storage_discharge += discharge
        
        # Compute power flow
        power_flow_result = compute_dc_power_flow(
            b_copy, l_copy, g_copy,
            renewable_factors={}  # Simplified: volatility handled separately
        )
        
        # Count statistics
        total_load_served = sum(b.load for b in b_copy if not b.disconnected)
        lines_tripped = [l.id for l in l_copy if l.tripped]
        buses_blacked_out = [b.id for b in b_copy if b.disconnected]
        generation_avail = sum(g.current_output * g.availability for g in g_copy)
        
        history["timestep"].append(t)
        history["total_load_served"].append(total_load_served)
        history["lines_tripped"].append(len(lines_tripped))
        history["buses_blacked_out"].append(len(buses_blacked_out))
        history["generation_available"].append(generation_avail)
        history["storage_discharged"].append(storage_discharge)
        
        if progress:
            load_pct = (total_load_served / total_initial_load * 100) if total_initial_load > 0 else 0
            print(f"Timestep {t}: {total_load_served:.0f}/{total_initial_load:.0f} MW ({load_pct:.1f}% served), "
                  f"{len(lines_tripped)} lines tripped, {len(buses_blacked_out)} buses out")
        
        # Identify overloaded lines and trip them (protection system response)
        overload_threshold = 1.15  # Lines trip at 115% capacity
        newly_trippped = []
        for line in l_copy:
            if not line.tripped and line.overload_factor > overload_threshold:
                line.tripped = True
                newly_trippped.append(line.id)
        
        # Check for convergence (no new tripped lines)
        if not newly_trippped and t > 0:
            if progress:
                print(f"\n  System stabilized at timestep {t}")
            break
        
        # Small delay to show progression
        if progress and t < num_timesteps - 1:
            time.sleep(0.05)
    
    # Calculate final metrics
    final_load = history["total_load_served"][-1]
    initial_load = sum(b.load for b in buses)
    blackout_fraction = 1 - (final_load / initial_load) if initial_load > 0 else 0
    max_overload = max((l.overload_factor for l in l_copy), default=0)
    
    return {
        "history": history,
        "final_load_served": final_load,
        "initial_load": initial_load,
        "blackout_fraction": blackout_fraction,
        "max_overload_factor": max_overload,
        "final_tripped_lines": lines_tripped,
        "final_blackout_buses": buses_blacked_out,
        "timesteps_to_stabilize": len(history["timestep"]),
        "storage_remaining_soc": sum(s.soc for s in s_copy) if s_copy else 0,
    }

# ──────────────────────────────────────────────
#   N-k Contingency Analysis
# ──────────────────────────────────────────────

def run_nk_contingency_analysis(
    buses: List[Bus],
    lines: List[TransmissionLine],
    generators: List[Generator],
    storage: List[Storage],
    max_k: int = 3,
    num_random_trials: int = 100,
    progress: bool = True,
) -> Dict:
    """
    Run N-k contingency analysis: simulate k-component failures randomly.
    
    Returns statistics on blackout risk at each k level.
    """
    results = {k: {"blackout_prob": 0, "avg_blackout_size": 0, "worst_case": 0} for k in range(1, max_k + 1)}
    
    # Get all failure candidates
    candidates = [line.id for line in lines] + [gen.id for gen in generators]
    
    for k in range(1, max_k + 1):
        blackout_events = 0
        total_blackout_size = 0
        worst_blackout = 0
        
        for trial in range(num_random_trials):
            # Pick k random components to fail
            failed = random.sample(candidates, k)
            
            result = simulate_cascading_failure(
                buses, lines, generators, storage,
                initial_contingency=failed,
                num_timesteps=10,
                renewable_volatility=0.0,
                use_storage=True,
                progress=False
            )
            
            if result["blackout_fraction"] > 0.01:  # More than 1% blackout
                blackout_events += 1
                total_blackout_size += result["blackout_fraction"]
                worst_blackout = max(worst_blackout, result["blackout_fraction"])
        
        results[k]["blackout_prob"] = blackout_events / num_random_trials
        results[k]["avg_blackout_size"] = total_blackout_size / num_random_trials if num_random_trials > 0 else 0
        results[k]["worst_case"] = worst_blackout
        
        if progress:
            print(f"N-{k}: {results[k]['blackout_prob']*100:.1f}% chance of blackout, "
                  f"avg size {results[k]['avg_blackout_size']*100:.1f}% of load")
    
    return results

# ──────────────────────────────────────────────
#   Renewable Intermittency Stress Test
# ──────────────────────────────────────────────

def test_renewable_intermittency(
    buses: List[Bus],
    lines: List[TransmissionLine],
    generators: List[Generator],
    storage: List[Storage],
    volatility_range: List[float],
    num_trials: int = 50,
    progress: bool = True,
) -> Dict:
    """
    Test grid stability under varying renewable volatility.
    
    Returns relationship between volatility and blackout risk.
    """
    results = []
    
    for volatility in volatility_range:
        blackout_risk = 0
        avg_load_lost = 0
        
        for trial in range(num_trials):
            # Start with random component failure
            candidate_lines = [l.id for l in lines]
            initial_failure = random.choice(candidate_lines)
            
            result = simulate_cascading_failure(
                buses, lines, generators, storage,
                initial_contingency=[initial_failure],
                num_timesteps=10,
                renewable_volatility=volatility,
                use_storage=True,
                progress=False
            )
            
            if result["blackout_fraction"] > 0:
                blackout_risk += 1
                avg_load_lost += result["blackout_fraction"]
        
        blackout_risk /= num_trials
        avg_load_lost /= num_trials
        
        results.append({
            "volatility": volatility,
            "blackout_probability": blackout_risk,
            "average_load_lost_pct": avg_load_lost * 100,
        })
        
        if progress:
            print(f"Volatility {volatility:.2f}: {blackout_risk*100:.1f}% blackout risk, "
                  f"avg loss {avg_load_lost*100:.1f}%")
    
    return results

# ──────────────────────────────────────────────
#   Visualization & Reporting
# ──────────────────────────────────────────────

def print_cascading_failure_report(result: Dict, scenario_name: str):
    """Print detailed report of a cascading failure simulation."""
    print(f"\n{'═' * 60}")
    print(f"  CASCADE EVENT REPORT: {scenario_name}")
    print(f"{'═' * 60}")
    
    history = result["history"]
    
    print(f"\n── Timeline of Blackout Progression ─────────────────────────────────")
    print(f"  {'T':>3} │ {'Load Served':>12} │ {'Lines Tripped':>13} │ {'Buses Out':>11}")
    print(f"  {'─' * 51}")
    
    for i, t in enumerate(history["timestep"]):
        load = history["total_load_served"][i]
        lines = history["lines_tripped"][i]
        buses = history["buses_blacked_out"][i]
        print(f"  {t:>3} │ {load:>10,.0f} MW │ {lines:>11} │ {buses:>10}")
    
    print(f"\n── Final Outcome ───────────────────────────────────────────────────")
    print(f"  Initial load:          {result['initial_load']:,.0f} MW")
    print(f"  Final load served:     {result['final_load_served']:,.0f} MW")
    print(f"  Total blackout:        {result['blackout_fraction']*100:.1f}%")
    print(f"  Lines permanently out: {len(result['final_tripped_lines'])}")
    print(f"  Buses without power:   {len(result['final_blackout_buses'])}")
    print(f"  Steps to stabilize:    {result['timesteps_to_stabilize']}")
    
    if result["final_tripped_lines"]:
        print(f"\n  Failed components:")
        for tid in result["final_tripped_lines"][:10]:
            print(f"    • {tid}")
    
    print(f"\n  Severity rating:", end=" ")
    if result["blackout_fraction"] < 0.05:
        print("MINOR (localized disruption)")
    elif result["blackout_fraction"] < 0.15:
        print("MODERATE (significant area affected)")
    elif result["blackout_fraction"] < 0.30:
        print("SEVERE (major regional blackout)")
    else:
        print("CATASTROPHIC (system-wide collapse)")
    
    print(f"{'═' * 60}\n")

def print_nk_analysis_report(nk_results: Dict):
    """Print N-k contingency analysis summary."""
    print(f"\n{'═' * 60}")
    print(f"  N-K CONTINGENCY ANALYSIS SUMMARY")
    print(f"{'═' * 60}")
    
    print(f"\n  {'k':>3} │ {'Blackout Prob.':>14} │ {'Avg Blackout Size':>17} │ {'Worst Case':>12}")
    print(f"  {'─' * 60}")
    
    for k, data in nk_results.items():
        prob = f"{data['blackout_prob']*100:.1f}%"
        avg = f"{data['avg_blackout_size']*100:.1f}%"
        worst = f"{data['worst_case']*100:.1f}%"
        print(f"  {k:>3} │ {prob:>14} │ {avg:>17} │ {worst:>12}")
    
    print(f"\n  N-1 (single failure): {nk_results.get(1, {}).get('blackout_prob', 0)*100:.1f}% blackout risk")
    print(f"  N-2 (double failure): {nk_results.get(2, {}).get('blackout_prob', 0)*100:.1f}% blackout risk")
    print(f"  N-3 (triple failure): {nk_results.get(3, {}).get('blackout_prob', 0)*100:.1f}% blackout risk")
    
    if nk_results.get(1, {}).get('blackout_prob', 0) < 0.1:
        print(f"\n  ✓ Grid passes N-1 criterion (standard reliability requirement)")
    else:
        print(f"\n  ⚠ Grid FAILS N-1 criterion — needs infrastructure investment")
    
    print(f"{'═' * 60}\n")

def print_intermittency_impact_report(interp_results: List[Dict]):
    """Print renewable intermittency stress test results."""
    print(f"\n{'═' * 60}")
    print(f"  RENEWABLE INTERMITTENCY IMPACT ANALYSIS")
    print(f"{'═' * 60}")
    
    print(f"\n  {'Volatility σ':>12} │ {'Blackout Risk':>13} │ {'Avg Load Lost':>15}")
    print(f"  {'─' * 52}")
    
    for r in interp_results:
        vol = f"{r['volatility']:.2f}"
        risk = f"{r['blackout_probability']*100:.1f}%"
        lost = f"{r['average_load_lost_pct']:.1f}%"
        print(f"  {vol:>12} │ {risk:>13} │ {lost:>15}")
    
    # Identify threshold where risk becomes unacceptable
    critical_volatility = None
    for r in interp_results:
        if r["blackout_probability"] >= 0.2:  # 20% risk threshold
            critical_volatility = r["volatility"]
            break
    
    if critical_volatility:
        print(f"\n  ⚠ Critical volatility threshold: σ ≈ {critical_volatility:.2f}")
        print("    Beyond this, blackout risk exceeds 20%")
        print("    Recommendation: Increase storage or dispatchable backup capacity")
    
    print(f"{'═' * 60}\n")

# ──────────────────────────────────────────────
#   CSV Export
# ──────────────────────────────────────────────

def export_cascade_csv(result: Dict, scenario_name: str, filename: str):
    """Export cascading failure history to CSV."""
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestep", "Load_Served_MW", "Lines_Tripped", 
                        "Buses_Blacked_Out", "Generation_Available", "Storage_Discharged"])
        history = result["history"]
        for i, t in enumerate(history["timestep"]):
            writer.writerow([
                t,
                history["total_load_served"][i],
                history["lines_tripped"][i],
                history["buses_blacked_out"][i],
                history["generation_available"][i],
                history["storage_discharged"][i],
            ])
    print(f"\n  Results exported to '{filename}'\n")

# ──────────────────────────────────────────────
#   CLI
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Power Grid Reliability Simulator — Cascading Failure Risk Assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python grid_relability.py                                   # Interactive\n"
            "  python grid_reliability.py --cascade -f T2                  # Single line failure\n"
            "  python grid_reliability.py --nk --grid renewable            # N-k on renewable grid\n"
            "  python grid_reliability.py --intermittency --storage        # Test with batteries\n"
            "  python grid_reliability.py --cascade -f T2,G1 --csv result.csv\n"
            "\n"
            "Recent real-world context:\n"
            "  • Iraq blackout Aug 2025: 6000 MW lost in seconds\n"
            "  • Iberia Aug 2025: Desynchronized from Continental Europe\n"
            "  • NERC Summer 2026: MISO, PJM, ERCOT high-risk regions\n"
        ),
    )
    parser.add_argument("--cascade", action="store_true",
                        help="Run single cascading failure scenario")
    parser.add_argument("-f", "--failure", type=str, nargs="+",
                        help="Specific components to fail (e.g., T2 G1)")
    parser.add_argument("--nk", action="store_true",
                        help="Run N-k contingency analysis")
    parser.add_argument("--intermittency", action="store_true",
                        help="Test renewable intermittency stress")
    parser.add_argument("--grid", type=str, default="simple",
                        choices=["simple", "renewable", "stressed"],
                        help="Grid topology (default: simple)")
    parser.add_argument("--storage", action="store_true",
                        help="Include battery storage")
    parser.add_argument("--steps", type=int, default=10,
                        help="Number of cascade steps to simulate")
    parser.add_argument("--csv", metavar="FILE", default=None,
                        help="Export results to CSV")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress messages")
    return parser.parse_args()

def interactive_mode():
    print("=" * 60)
    print("  POWER GRID RELIABILITY SIMULATOR")
    print("  Cascading Failure Risk Assessment")
    print("=" * 60)
    print()
    print("  Recent context:")
    print("  • Iraq Aug 2025: 6000 MW blackout in seconds")
    print("  • Iberia Aug 2025: Desynchronized from Europe")
    print("  • NERC Summer 2026: High-risk regions identified")
    print()
    
    print("  Select grid configuration:")
    print("    1. Simple 5-bus test system")
    print("    2. Renewable-heavy grid (high intermittency)")
    print("    3. Stressed grid (NERC high-risk scenario)")
    
    while True:
        grid_choice = input("  Selection [1]: ").strip() or "1"
        if grid_choice in ("1", "2", "3"):
            break
        print("  ⚠  Please enter 1, 2, or 3.")
    
    print("\n  Select analysis mode:")
    print("    1. Cascading failure simulation")
    print("    2. N-k contingency analysis")
    print("    3. Renewable intermittency stress test")
    
    while True:
        mode_choice = input("  Selection [1]: ").strip() or "1"
        if mode_choice in ("1", "2", "3"):
            break
        print("  ⚠  Please enter 1, 2, or 3.")
    
    return int(grid_choice), int(mode_choice)

# ──────────────────────────────────────────────
#   Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()
    progress = not args.quiet
    
    # Build grid
    if args.grid == "simple":
        buses, lines, generators, storage = build_simple_grid()
    elif args.grid == "renewable":
        buses, lines, generators, storage = build_renewable_heavy_grid()
        if not args.storage:
            storage = []  # Override if explicitly disabled
    else:  # stressed
        buses, lines, generators, storage = build_stressed_grid()
        storage = []  # Stressed grid has no storage
    
    results = []
    
    if args.cascade or (not (args.cascade or args.nk or args.intermittency)):
        # Determine failure set
        if args.failure:
            failures = args.failure
        else:
            # Pick random failure for demo
            all_components = [l.id for l in lines] + [g.id for g in generators]
            failures = [random.choice(all_components)]
        
        grid_name = {"simple": "Simple 5-Bus", "renewable": "Renewable Heavy", "stressed": "Stressed NERC"}
        scenario_name = f"{grid_name[args.grid]} + {failures}"
        
        result = simulate_cascading_failure(
            buses, lines, generators, storage,
            initial_contingency=failures,
            num_timesteps=args.steps,
            renewable_volatility=0.3 if args.grid == "renewable" else 0.0,
            use_storage=args.storage,
            progress=progress
        )
        
        print_cascading_failure_report(result, scenario_name)
        results.append(("cascade", result))
        
        if args.csv:
            export_cascade_csv(result, scenario_name, args.csv)
    
    if args.nk or (not (args.cascade or args.nk or args.intermittency) and interactive_mode()[1] == 2):
        nk_results = run_nk_contingency_analysis(
            buses, lines, generators, storage,
            max_k=3,
            num_random_trials=50,
            progress=progress
        )
        print_nk_analysis_report(nk_results)
        results.append(("nk", nk_results))
    
    if args.intermittency:
        interp_results = test_renewable_intermittency(
            buses, lines, generators, storage,
            volatility_range=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            num_trials=50,
            progress=progress
        )
        print_intermittency_impact_report(interp_results)
        results.append(("intermittency", interp_results))
    
    if not (args.cascade or args.nk or args.intermittency):
        print("\n💡 Tips:")
        print("  • Run with --cascade -f T2 to test specific line failure")
        print("  • Run with --nk --grid renewable to see how renewables affect N-k risk")
        print("  • Run with --intermittency --storage to quantify battery benefits")
        print("  • Use --grid stressed for near-collapse scenario analysis")

if __name__ == "__main__":
    main()