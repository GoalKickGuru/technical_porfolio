"""
Market Entry & Competitive Positioning Simulator
===============================================
Adapting Rock-Paper-Scissors game theory to business strategy
for market entry decisions and competitive positioning.

Strategies & Payoff Matrix:
- Price Leadership beats Niche (capture volume)
- Niche beats Differentiation (specialized focus)
- Differentiation beats Innovation (brand loyalty)
- Innovation beats Price Leadership (first-mover advantage)
- Cost Leadership beats Premium (efficiency wins)

Cyclic dominance mirrors real market dynamics where no single
strategy dominates all competitors—creating a Nash equilibrium.
"""

import random, time, sys, json, os
from collections import defaultdict, Counter
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

# ============================================================================
# STRATEGY DEFINITIONS
# ============================================================================

STRATEGIES = {
    'PRICE_LEAD': '🏷️ Price Leadership',
    'NICHE': '🎯 Niche Focus',
    'DIFFERENTIATION': '✨ Differentiation',
    'INNOVATION': '💡 Innovation',
    'COST_LEAD': '⚙️ Cost Optimization'
}

STRATEGY_DESCRIPTORS = {
    'PRICE_LEAD': '''Target mass market with lowest prices.
    Pros: High volume, quick market share
    Cons: Thin margins, vulnerable to cost leaders''',
    
    'NICHE': '''Focus on underserved specialized segment.
    Pros: High margins, loyal customers
    Cons: Limited ceiling, easy to overlook''',
    
    'DIFFERENTIATION': '''Stand out via unique features/brand.
    Pros: Premium pricing, sticky customers
    Cons: Requires marketing spend, imitation risk''',
    
    'INNOVATION': '''First-to-market new products/services.
    Pros: First-mover advantage, patent protection
    Cons: High R&D costs, adoption risk''',
    
    'COST_LEAD': '''Operational efficiency drives lower costs.
    Pros: Sustainable advantage, margin buffer
    Cons: Heavy upfront investment, execution risk'''
}

# ============================================================================
# GAME THEORY PAYOFF MATRIX
# ============================================================================

# Who beats whom - cyclic dominance relationship
BEATS = {
    'PRICE_LEAD': ['DIFFERENTIATION'],  # Low price undercuts premium brands
    'NICHE': ['COST_LEAD'],              # Specialized focus beats generic efficiency
    'DIFFERENTIATION': ['INNOVATION'],   # Established brand beats new entrant
    'INNOVATION': ['PRICE_LEAD'],        # First-mover advantage beats price war
    'COST_LEAD': ['PRICE_LEAD']          # Lower costs sustain below-market pricing
}

# Detailed payoff matrix: (player_payoff, opponent_payoff)
PAYOFF_MATRIX = {
    ('PRICE_LEAD', 'PRICE_LEAD'): (-2, -2),     # Price war hurts both
    ('PRICE_LEAD', 'NICHE'): (3, -1),           # Volume captures niche attention
    ('PRICE_LEAD', 'DIFFERENTIATION'): (4, -2), # Undercut premium brand
    ('PRICE_LEAD', 'INNOVATION'): (-1, 3),      # New product commands premium
    ('PRICE_LEAD', 'COST_LEAD'): (-3, 3),       # Cannot compete with lower costs
    
    ('NICHE', 'PRICE_LEAD'): (-1, 3),           # Niche loses volume battle
    ('NICHE', 'NICHE'): (0, 0),                 # Direct rivalry, split market
    ('NICHE', 'DIFFERENTIATION'): (2, 1),       # Both serve distinct segments
    ('NICHE', 'INNOVATION'): (1, 2),            # Innovation attracts broader audience
    ('NICHE', 'COST_LEAD'): (-2, 4),            # Generic cheaper alternative wins
    
    ('DIFFERENTIATION', 'PRICE_LEAD'): (-2, 4), # Undercut by low-price leader
    ('DIFFERENTIATION', 'NICHE'): (1, 2),       # Niche retains specialized appeal
    ('DIFFERENTIATION', 'DIFFERENTIATION'): (-1, -1), # Brand wars, no clear winner
    ('DIFFERENTIATION', 'INNOVATION'): (3, -2), # Established brand trusted over new
    ('DIFFERENTIATION', 'COST_LEAD'): (0, 2),   # Efficiency gains matter more
    
    ('INNOVATION', 'PRICE_LEAD'): (3, -1),      # New product commands premium
    ('INNOVATION', 'NICHE'): (2, 1),            # Broader appeal than narrow focus
    ('INNOVATION', 'DIFFERENTIATION'): (-2, 3), # Established brand wins loyalty
    ('INNOVATION', 'INNOVATION'): (1, 1),       # Co-innovation, shared growth
    ('INNOVATION', 'COST_LEAD'): (-1, 2),       # Efficiency beats new unproven tech
    
    ('COST_LEAD', 'PRICE_LEAD'): (3, -3),       # Sustainable below-market pricing
    ('COST_LEAD', 'NICHE'): (4, -2),            # Cheaper alternatives attract mass
    ('COST_LEAD', 'DIFFERENTIATION'): (2, 0),   # Value-conscious switch to cheaper
    ('COST_LEAD', 'INNOVATION'): (2, -1),       # Efficiency proven over new tech
    ('COST_LEAD', 'COST_LEAD'): (0, 0),         # Efficiencies cancel out
}

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class MarketOutcome:
    round_number: int
    player_strategy: str
    competitor_strategy: str
    player_payoff: int
    competitor_payoff: int
    market_share_change: float
    timestamp: str
    
    def to_dict(self):
        return asdict(self)

@dataclass
class FirmState:
    name: str
    type: str  # 'player' or 'ai'
    capital: float
    market_share: float
    reputation_score: float
    strategy_history: List[str]
    cumulative_payoff: int
    wins: int
    losses: int
    ties: int
    streak: int
    
    def __init__(self, name: str, firm_type: str = 'player'):
        self.name = name
        self.type = firm_type
        self.capital = 1000000.0  # $1M starting capital
        self.market_share = 15.0  # 15% initial market share
        self.reputation_score = 50.0  # 0-100 scale
        self.strategy_history = []
        self.cumulative_payoff = 0
        self.wins = 0
        self.losses = 0
        self.ties = 0
        self.streak = 0
    
    def update_after_round(self, payoff: int, won: bool, lost: bool):
        self.cumulative_payoff += payoff
        self.capital += payoff * 1000  # Scale payoffs to dollars
        self.strategy_history.append({'payoff': payoff, 'timestamp': datetime.now().isoformat()})
        
        if won:
            self.wins += 1
            self.streak += 1
            self.market_share += abs(payoff) * 0.02
            self.reputation_score = min(100, self.reputation_score + 2)
        elif lost:
            self.losses += 1
            self.streak = 0
            self.market_share -= abs(payoff) * 0.02
            self.reputation_score = max(0, self.reputation_score - 2)
        else:
            self.ties += 1
            self.streak = 0
        
        self.market_share = max(5.0, min(60.0, self.market_share))  # Bounds


# ============================================================================
# AI OPPONENT WITH MARKET INTELLIGENCE
# ============================================================================

class CompetitorAI:
    """
    AI competitor that analyzes player patterns and adjusts strategy.
    Models how real competitors use market intelligence.
    """
    
    def __init__(self, name: str = "Competitor", difficulty: str = 'medium'):
        self.name = name
        self.difficulty = difficulty
        self.player_moves_history = []
        self.strategy_performance = defaultdict(list)  # Track which strategies beat what
        
    def get_counter_strategy(self, player_move: str) -> str:
        """Find strategy that beats the player's move"""
        counter_map = {v: k for k, values in BEATS.items() for v in values}
        return counter_map.get(player_move, random.choice(list(STRATEGIES.keys())))
    
    def analyze_patterns(self) -> Dict[str, float]:
        """Calculate player's strategy tendencies"""
        if not self.player_moves_history:
            return {s: 0.2 for s in STRATEGIES.keys()}
        
        counts = Counter(self.player_moves_history[-10:])  # Last 10 moves
        total = len(counts)
        tendencies = {s: counts.get(s, 0) / total for s in STRATEGIES.keys()}
        return tendencies
    
    def predict_next_move(self) -> str:
        """Predict player's likely next strategy"""
        if len(self.player_moves_history) < 3:
            return random.choice(list(STRATEGIES.keys()))
        
        tendencies = self.analyze_patterns()
        
        # Check for streak behavior
        recent = self.player_moves_history[-3:]
        if recent[0] == recent[1] == recent[2]:
            # Player repeating same strategy (risk-seeking)
            return random.choice(list(STRATEGIES.keys()))
        
        # Check for recency bias
        most_recent = self.player_moves_history[-1]
        counter = self.get_counter_strategy(most_recent)
        
        return counter
    
    def select_strategy(self, player_last_move: str = None) -> str:
        """Main strategy selection based on difficulty"""
        if player_last_move:
            self.player_moves_history.append(player_last_move)
        
        if self.difficulty == 'easy':
            return random.choice(list(STRATEGIES.keys()))
        
        elif self.difficulty == 'medium':
            # 50% chance to counter, 50% random
            if random.random() < 0.5 and player_last_move:
                return self.get_counter_strategy(player_last_move)
            return random.choice(list(STRATEGIES.keys()))
        
        elif self.difficulty == 'hard':
            # Analyze patterns, exploit tendencies
            tendencies = self.analyze_patterns()
            predicted = self.predict_next_move()
            
            # Counter the predicted move
            return self.get_counter_strategy(predicted)
        
        return random.choice(list(STRATEGIES.keys()))


# ============================================================================
# MARKET SIMULATION ENGINE
# ============================================================================

class MarketSimulation:
    """Core simulation engine for competitive market dynamics"""
    
    def __init__(self, num_competitors: int = 2, difficulty: str = 'medium'):
        self.num_competitors = num_competitors
        self.ai_opponents = [CompetitorAI(name=f"Competitor_{i+1}", difficulty=difficulty) 
                            for i in range(num_competitors)]
        self.player_firm = FirmState(name="Your Firm", firm_type='player')
        self.round_history: List[MarketOutcome] = []
        self.simulation_active = True
        self.current_market_size = 100000000.0  # $100M TAM
        self.inflation_rate = 0.02  # 2% per round
        
    def resolve_outcome(self, player_strategy: str, opponent_strategy: str) -> Tuple[int, int]:
        """Resolve single round using payoff matrix"""
        key = (player_strategy, opponent_strategy)
        if key in PAYOFF_MATRIX:
            return PAYOFF_MATRIX[key]
        
        # Fallback for unknown combinations
        reverse_key = (opponent_strategy, player_strategy)
        if reverse_key in PAYOFF_MATRIX:
            opp_p, plyr_p = PAYOFF_MATRIX[reverse_key]
            return plyr_p, opp_p
        
        return (0, 0)  # Neutral outcome
        
    def run_round(self, player_strategy: str) -> MarketOutcome:
        """Execute one round of market competition"""
        # Get opponent strategy
        opponent = self.ai_opponents[0]
        opponent_strategy = opponent.select_strategy(player_strategy)
        
        # Resolve outcomes
        player_payoff, opponent_payoff = self.resolve_outcome(
            player_strategy, opponent_strategy
        )
        
        # Calculate market share impact
        payoff_diff = player_payoff - opponent_payoff
        market_share_delta = payoff_diff * 0.5
        
        # Update firm states
        self.player_firm.update_after_round(
            player_payoff, 
            won=player_payoff > opponent_payoff,
            lost=player_payoff < opponent_payoff
        )
        # Note: Could track AI firm state similarly
        
        # Create outcome record
        outcome = MarketOutcome(
            round_number=len(self.round_history) + 1,
            player_strategy=player_strategy,
            competitor_strategy=opponent_strategy,
            player_payoff=player_payoff,
            competitor_payoff=opponent_payoff,
            market_share_change=market_share_delta,
            timestamp=datetime.now().isoformat()
        )
        
        self.round_history.append(outcome)
        return outcome
    
    def simulate_tournament(self, player_strategy_sequence: List[str]) -> Dict:
        """Run multi-round tournament simulation"""
        results = []
        cumulative_payoff = 0
        
        for strategy in player_strategy_sequence:
            outcome = self.run_round(strategy)
            results.append(outcome)
            cumulative_payoff += outcome.player_payoff
        
        return {
            'total_rounds': len(results),
            'cumulative_payoff': cumulative_payoff,
            'wins': sum(1 for o in results if o.player_payoff > o.competitor_payoff),
            'losses': sum(1 for o in results if o.player_payoff < o.competitor_payoff),
            'ties': sum(1 for o in results if o.player_payoff == o.competitor_payoff),
            'final_market_share': self.player_firm.market_share
        }
    
    def get_strategy_recommendation(self) -> str:
        """Analyze history and recommend best strategy"""
        if not self.round_history:
            return random.choice(list(STRATEGIES.keys()))
        
        strategy_perf = defaultdict(int)
        for outcome in self.round_history:
            payoff = outcome.player_payoff
            strategy_perf[outcome.player_strategy] += payoff
        
        if not strategy_perf:
            return random.choice(list(STRATEGIES.keys()))
        
        best_strategy = max(strategy_perf, key=strategy_perf.get)
        return best_strategy


# ============================================================================
# USER INTERFACE
# ============================================================================

class MarketStrategyGame:
    """Main game interface"""
    
    def __init__(self):
        self.sim = MarketSimulation(difficulty='medium')
        self.running = True
        
    def display_banner(self):
        """Show welcome banner"""
        print("\n" + "="*70)
        print("  🏢 MARKET ENTRY & COMPETITIVE POSITIONING SIMULATOR 🏢")
        print("  Game Theory Applied to Business Strategy")
        print("="*70 + "\n")
    
    def display_strategies(self):
        """Show available strategies with descriptions"""
        print("\n📋 AVAILABLE STRATEGIES:")
        print("-"*70)
        for code, name in STRATEGIES.items():
            desc = STRATEGY_DESCRIPTORS[code][:60] + "..."
            print(f"  [{code[:4]}] {name}")
            print(f"       {desc}")
            print()
        print("-"*70 + "\n")
    
    def display_payoff_matrix(self):
        """Show simplified payoff relationships"""
        print("\n🎲 STRATEGY INTERACTIONS (Who Beats Whom):")
        print("-"*50)
        for strategy, counters in BEATS.items():
            for counter in counters:
                print(f"  {STRATEGY_STRINGS.get(strategy, strategy)} → {STRATEGY_STRINGS.get(counter, counter)}")
        print("-"*50 + "\n")
    
    def display_dashboard(self):
        """Show current firm status"""
        fs = self.sim.player_firm
        
        print("\n" + "="*70)
        print("  YOUR FIRM STATUS")
        print("="*70)
        print(f"  💵 Capital:        ${fs.capital:,.0f}")
        print(f"  📊 Market Share:   {fs.market_share:.1f}%")
        print(f"  ⭐ Reputation:      {fs.reputation_score:.0f}/100")
        print(f"  🔥 Win Streak:     {fs.streak}")
        print()
        print(f"  📈 Total Rounds:   {len(self.sim.round_history)}")
        print(f"  ✅ Wins:           {fs.wins}")
        print(f"  ❌ Losses:         {fs.losses}")
        print(f"  🤝 Ties:           {fs.ties}")
        print(f"  💰 Cumulative:     {'+' if fs.cumulative_payoff >= 0 else ''}{fs.cumulative_payoff}")
        print("="*70 + "\n")
    
    def display_main_menu(self):
        """Show main navigation menu"""
        print("╔══════════════════════════════════════════════════════════╗")
        print("║                    MAIN MENU                             ║")
        print("╠══════════════════════════════════════════════════════════╣")
        print("║  1. ▶ Single Round Competition                           ║")
        print("║  2. 🎯 Strategy Analysis                                 ║")
        print("║  3. 🏆 Tournament Mode (Best of 7)                       ║")
        print("║  4. 📊 View Payoff Matrix                                ║")
        print("║  5. 📜 Strategy Descriptions                             ║")
        print("║  6. ⚙️ Simulation Settings                               ║")
        print("║  7. 💾 Save & Exit                                       ║")
        print("║  8. ℹ️ About                                             ║")
        print("╚══════════════════════════════════════════════════════════╝")
    
    def play_single_round(self):
        """Play one round of market competition"""
        self.display_strategies()
        
        # Get player input
        player_strategy = self.get_valid_strategy_input()
        if player_strategy is None:
            return
            
        # Dramatic tension builder
        print("\n🔄 Submitting strategy...")
        time.sleep(1)
        print("📡 Monitoring competitor intelligence...")
        time.sleep(1)
        print("💼 Market dynamics resolving...")
        time.sleep(1.5)
        
        # Run round
        outcome = self.sim.run_round(player_strategy)
        
        # Display results
        print("\n" + "="*70)
        print("  ROUND RESULTS")
        print("="*70)
        print(f"  Your Strategy:      {STRATEGIES.get(outcome.player_strategy, 'Unknown')}")
        print(f"  Competitor Strategy:{STRATEGIES.get(outcome.competitor_strategy, 'Unknown')}")
        print()
        
        if outcome.player_payoff > outcome.competitor_payoff:
            print(f"  🎉 VICTORY! You gained {outcome.player_payoff} points")
            print(f"  📈 Market Share: {outcome.market_share_change:+.2f}%")
        elif outcome.player_payoff < outcome.competitor_payoff:
            print(f"  😞 DEFEAT. You lost {abs(outcome.player_payoff)} points")
            print(f"  📉 Market Share: {outcome.market_share_change:+.2f}%")
        else:
            print(f"  🤝 STALEMATE. Payoffs equal")
        
        print("="*70 + "\n")
        time.sleep(2)
    
    def play_tournament(self):
        """Run tournament mode - best of 7 rounds"""
        print("\n🏆 ENTERING TOURNAMENT MODE")
        print("Best of 7 rounds - first to win 4 takes the championship!\n")
        time.sleep(2)
        
        player_wins = 0
        comp_wins = 0
        rounds_needed = 4
        outcomes_list = []
        
        for round_num in range(7):
            if player_wins >= rounds_needed or comp_wins >= rounds_needed:
                break
                
            print(f"\n--- Round {round_num + 1} ---")
            self.display_strategies()
            
            player_strategy = self.get_valid_strategy_input()
            if player_strategy is None:
                break
            
            outcome = self.sim.run_round(player_strategy)
            outcomes_list.append(outcome)
            
            if outcome.player_payoff > outcome.competitor_payoff:
                player_wins += 1
                symbol = "✅"
            elif outcome.player_payoff < outcome.competitor_payoff:
                comp_wins += 1
                symbol = "❌"
            else:
                symbol = "🤝"
            
            print(f"{symbol} Score: {player_wins} - {comp_wins}\n")
            time.sleep(1)
        
        # Championship result
        print("\n" + "="*70)
        print("  🏆 TOURNAMENT CHAMPIONSHIP RESULT 🏆")
        print("="*70)
        
        if player_wins >= rounds_needed:
            print(f"  🎉 CONGRATULATIONS! You won {player_wins}-{comp_wins}")
            print(f"  💰 Championship Bonus: +${abs(sum(o.player_payoff for o in outcomes_list))*100:,.0f}")
        elif comp_wins >= rounds_needed:
            print(f"  💀 Defeated {comp_wins}-{player_wins}")
            print(f"  Market share lost to aggressive competitor")
        else:
            print(f"  🤝 Tournament ended {player_wins}-{comp_wins} (Draw)")
        
        print("="*70 + "\n")
        time.sleep(3)
    
    def get_valid_strategy_input(self) -> Optional[str]:
        """Get and validate player strategy choice"""
        valid_inputs = list(STRATEGIES.keys()) + ['Q']
        
        while True:
            print("Enter your strategy code (or Q to quit): ", end="")
            choice = input().strip().upper()
            
            if choice == 'Q':
                return None
            
            if choice in valid_inputs:
                return choice
            
            print("❌ Invalid code. Try again.\n")
    
    def run(self):
        """Main game loop"""
        self.display_banner()
        time.sleep(1)
        
        while self.running:
            self.display_main_menu()
            selection = input("\nSelect option: ").strip()
            
            if selection == '1':
                self.play_single_round()
                
            elif selection == '2':
                self.sim.get_strategy_recommendation()
                print("\n💡 Recommended: Use our analysis to adjust strategy based on\n     competitor patterns. See tournament mode for deeper insights.")
                time.sleep(3)
                
            elif selection == '3':
                self.play_tournament()
                
            elif selection == '4':
                self.display_payoff_matrix()
                input("Press Enter to continue...")
                
            elif selection == '5':
                self.display_strategies()
                input("Press Enter to continue...")
                
            elif selection == '6':
                print("\n⚙️ SETTINGS")
                print("1. Change Difficulty (Easy/Medium/Hard)")
                print("2. Adjust Number of Competitors")
                print("3. Reset Progress")
                print("4. Back to Menu")
                
                setting = input("Option: ").strip()
                if setting == '1':
                    diff = input("Difficulty [easy/medium/hard]: ").strip().lower()
                    self.sim = MarketSimulation(difficulty=diff if diff in ['easy','medium','hard'] else 'medium')
                    print("✓ Difficulty updated!")
                elif setting == '2':
                    num = input("Competitors [1-3]: ").strip()
                    try:
                        self.sim = MarketSimulation(num_competitors=int(num), 
                                                   difficulty=self.sim.ai_opponents[0].difficulty)
                        print("✓ Competitors updated!")
                    except:
                        print("Invalid input")
                elif setting == '3':
                    confirm = input("Reset all progress? (yes/no): ").strip().lower()
                    if confirm == 'yes':
                        self.sim = MarketSimulation()
                        print("✓ Progress reset!")
                        
                input("Press Enter to continue...")
                
            elif selection == '7':
                self.sim.save_game()
                print("💾 Game saved. Thanks for playing!")
                self.running = False
                
            elif selection == '8':
                print("\n📘 ABOUT")
                print("="*50)
                print("This simulator applies game theory to business strategy.")
                print("Cyclic dominance (like RPS) reflects real markets where")
                print("no single strategy beats all others—a Nash equilibrium.")
                print()
                print("Payoffs represent relative competitive advantage.")
                print("Higher capital = larger position in the market.")
                print("="*50)
                input("Press Enter to continue...")
                
            else:
                print("❌ Invalid selection. Try again.")


# ============================================================================
# HELPER FUNCTIONS & CONSTANTS
# ============================================================================

STRATEGY_STRINGS = {
    'PRICE_LEAD': 'Price Leadership',
    'NICHE': 'Niche Focus',
    'DIFFERENTIATION': 'Differentiation',
    'INNOVATION': 'Innovation',
    'COST_LEAD': 'Cost Leadership'
}


def main():
    """Entry point"""
    try:
        game = MarketStrategyGame()
        game.run()
    except KeyboardInterrupt:
        print("\n\n⛔ Game interrupted. Closing safely.")
        sys.exit()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()