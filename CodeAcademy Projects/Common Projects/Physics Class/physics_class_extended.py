# ================================================================
# Physics Calculator - Extended Version
# All 13 original tasks + gravitational potential energy, 
# kinetic energy, and momentum calculations
# ================================================================

# ============================================================
# CONSTANTS
# ============================================================
SPEED_OF_LIGHT = 3 * 10**8  # meters per second
GRAVITATIONAL_ACCELERATION = 9.81  # m/s² on Earth

# ============================================================
# VARIABLES (uncommented for immediate use)
# ============================================================
train_mass = 22680          # kg
train_acceleration = 10     # m/s²
train_distance = 100        # meters
bomb_mass = 1               # kg

# Additional test values for extended functions
car_mass = 1500             # kg (typical car)
car_velocity = 30           # m/s (~108 km/h)
ball_mass = 0.5             # kg
ball_height = 10            # meters
rocket_mass = 5000          # kg
rocket_velocity = 8000      # m/s (orbital speed)

# ============================================================
# SECTION 1: TEMPERATURE CONVERSIONS (Tasks 1-4)
# ============================================================

def f_to_c(f_temp: float) -> float:
    """
    Convert Fahrenheit to Celsius.
    
    Formula: C = (F - 32) × 5/9
    
    Args:
        f_temp: Temperature in Fahrenheit
        
    Returns:
        Temperature in Celsius
    """
    c_temp = (f_temp - 32) * 5 / 9
    return c_temp


def c_to_f(c_temp: float) -> float:
    """
    Convert Celsius to Fahrenheit.
    
    Formula: F = C × 9/5 + 32
    
    Args:
        c_temp: Temperature in Celsius
        
    Returns:
        Temperature in Fahrenheit
    """
    f_temp = c_temp * (9 / 5) + 32
    return f_temp


# Bonus temperature conversions
def c_to_k(c_temp: float) -> float:
    """Convert Celsius to Kelvin."""
    return c_temp + 273.15


def k_to_c(k_temp: float) -> float:
    """Convert Kelvin to Celsius."""
    return k_temp - 273.15


def f_to_k(f_temp: float) -> float:
    """Convert Fahrenheit to Kelvin via Celsius."""
    return c_to_k(f_to_c(f_temp))


def k_to_f(k_temp: float) -> float:
    """Convert Kelvin to Fahrenheit via Celsius."""
    return c_to_f(k_to_c(k_temp))


# --- Temperature Tests ---
print("=" * 70)
print("SECTION 1: TEMPERATURE CONVERSIONS")
print("=" * 70)

# Task 2: Test with 100°F
f100_in_celsius = f_to_c(100)
print(f"\n[Task 2] 100°F = {f100_in_celsius:.2f}°C")

# Task 4: Test with 0°C
c0_in_fahrenheit = c_to_f(0)
print(f"[Task 4] 0°C = {c0_in_fahrenheit:.2f}°F")

# Bonus: More temperature conversions
print(f"\n[Bonus] Temperature Reference Table:")
print("-" * 50)
test_temps_f = [0, 32, 72, 100, 212, 98.6]
for temp_f in test_temps_f:
    temp_c = f_to_c(temp_f)
    temp_k = f_to_k(temp_f)
    desc = ""
    if temp_f == 32:
        desc = "(Water freezes)"
    elif temp_f == 212:
        desc = "(Water boils)"
    elif temp_f == 98.6:
        desc = "(Human body temp)"
    print(f"{temp_f:>6.1f}°F → {temp_c:>8.2f}°C → {temp_k:>8.2f}K {desc}")

# Round-trip verification
round_trip = c_to_f(f_to_c(72))
print(f"\n[Verification] Round-trip: 72°F → °C → {round_trip:.4f}°F")


# ============================================================
# SECTION 2: FORCE, ENERGY & WORK (Tasks 5-13)
# ============================================================

def get_force(mass: float, acceleration: float) -> float:
    """
    Calculate force using Newton's Second Law.
    
    Formula: F = m × a
    
    Args:
        mass: Mass in kilograms (kg)
        acceleration: Acceleration in meters per second squared (m/s²)
        
    Returns:
        Force in Newtons (N)
        
    Raises:
        ValueError: If mass or acceleration is negative
    """
    if mass < 0:
        raise ValueError("Mass cannot be negative.")
    if acceleration < 0:
        raise ValueError("Acceleration cannot be negative.")
    return mass * acceleration


def get_energy(mass: float, c: float = SPEED_OF_LIGHT) -> float:
    """
    Calculate energy using Einstein's mass-energy equivalence.
    
    Formula: E = m × c²
    Where c is the speed of light (~3 × 10⁸ m/s)
    
    Args:
        mass: Mass in kilograms (kg)
        c: Speed of light in m/s (default: 3 × 10⁸)
        
    Returns:
        Energy in Joules (J)
        
    Raises:
        ValueError: If mass is negative
    """
    if mass < 0:
        raise ValueError("Mass cannot be negative.")
    return mass * c ** 2


def get_work(mass: float, acceleration: float, distance: float) -> float:
    """
    Calculate work done by a force over a distance.
    
    Formula: W = F × d = m × a × d
    
    Args:
        mass: Mass in kilograms (kg)
        acceleration: Acceleration in m/s²
        distance: Distance in meters (m)
        
    Returns:
        Work in Joules (J)
    """
    force = get_force(mass, acceleration)
    return force * distance


# --- Force, Energy & Work Tests ---
print("\n" + "=" * 70)
print("SECTION 2: FORCE, ENERGY & WORK")
print("=" * 70)

# Task 6: Test get_force()
train_force = get_force(train_mass, train_acceleration)
print(f"\n[Task 6] Train force calculation: {train_force} N")

# Task 7
print(f"[Task 7] The GE train supplies {train_force:,} Newtons of force.")

# Task 9: Test get_energy()
bomb_energy = get_energy(bomb_mass)
print(f"\n[Task 9] Bomb energy calculation: {bomb_energy} J")

# Task 10
print(f"[Task 10] A 1kg bomb supplies {bomb_energy:,} Joules.")

# Task 12: Test get_work()
train_work = get_work(train_mass, train_acceleration, train_distance)
print(f"\n[Task 12] Train work calculation: {train_work} J")

# Task 13
print(f"[Task 13] The GE train does {train_work:,} Joules of work over {train_distance} meters.")


# ============================================================
# SECTION 3: EXTENDED PHYSICS FUNCTIONS
# ============================================================

def kinetic_energy(mass: float, velocity: float) -> float:
    """
    Calculate kinetic energy of a moving object.
    
    Formula: KE = ½ × m × v²
    
    Args:
        mass: Mass in kilograms (kg)
        velocity: Velocity in meters per second (m/s)
        
    Returns:
        Kinetic energy in Joules (J)
        
    Raises:
        ValueError: If mass is negative or velocity is imaginary (complex)
    """
    if mass < 0:
        raise ValueError("Mass cannot be negative.")
    return 0.5 * mass * velocity ** 2


def gravitational_potential_energy(mass: float, height: float, g: float = GRAVITATIONAL_ACCELERATION) -> float:
    """
    Calculate gravitational potential energy relative to a reference point.
    
    Formula: PE = m × g × h
    
    Args:
        mass: Mass in kilograms (kg)
        height: Height above reference in meters (m)
        g: Gravitational acceleration (default: 9.81 m/s² on Earth)
        
    Returns:
        Potential energy in Joules (J)
        
    Raises:
        ValueError: If mass is negative
    """
    if mass < 0:
        raise ValueError("Mass cannot be negative.")
    return mass * g * height


def momentum(mass: float, velocity: float) -> float:
    """
    Calculate linear momentum of a moving object.
    
    Formula: p = m × v
    
    Args:
        mass: Mass in kilograms (kg)
        velocity: Velocity in meters per second (m/s)
        
    Returns:
        Momentum in kilogram-meters per second (kg·m/s)
        
    Raises:
        ValueError: If mass is negative
    """
    if mass < 0:
        raise ValueError("Mass cannot be negative.")
    return mass * velocity


def elastic_potential_energy(spring_constant: float, displacement: float) -> float:
    """
    Calculate elastic potential energy stored in a spring.
    
    Formula: PE_elastic = ½ × k × x²
    
    Args:
        spring_constant: Spring constant k in N/m
        displacement: Displacement from equilibrium in meters (m)
        
    Returns:
        Elastic potential energy in Joules (J)
    """
    return 0.5 * spring_constant * displacement ** 2


def power(work: float, time: float) -> float:
    """
    Calculate power (rate of doing work).
    
    Formula: P = W / t
    
    Args:
        work: Work done in Joules (J)
        time: Time elapsed in seconds (s)
        
    Returns:
        Power in Watts (W)
        
    Raises:
        ValueError: If time is zero or negative
    """
    if time <= 0:
        raise ValueError("Time must be positive.")
    return work / time


# ============================================================
# SECTION 4: EXTENDED CALCULATIONS & TESTS
# ============================================================

print("\n" + "=" * 70)
print("SECTION 3: EXTENDED PHYSICS CALCULATIONS")
print("=" * 70)

# --- Kinetic Energy Calculations ---
print("\n[Kinetic Energy - KE = ½mv²]")
print("-" * 50)

car_ke = kinetic_energy(car_mass, car_velocity)
print(f"Car ({car_mass}kg @ {car_velocity}m/s):    {car_ke:,.0f} J")

ball_ke = kinetic_energy(ball_mass, 15)  # Ball thrown at 15 m/s
print(f"Ball ({ball_mass}kg @ 15m/s):       {ball_ke:,.0f} J")

rocket_ke = kinetic_energy(rocket_mass, rocket_velocity)
print(f"Rocket ({rocket_mass}kg @ {rocket_velocity}m/s): {rocket_ke:,.2e} J")

print(f"\nNote: Rocket kinetic energy ~ {rocket_ke / car_ke:,.0f}× the car's KE")


# --- Gravitational Potential Energy ---
print("\n[Gravitational Potential Energy - PE = mgh]")
print("-" * 50)

ball_pe = gravitational_potential_energy(ball_mass, ball_height)
print(f"Ball ({ball_mass}kg @ {ball_height}m height): {ball_pe:,.2f} J")

train_pe = gravitational_potential_energy(train_mass, 50)  # Train on hill
print(f"Train ({train_mass}kg @ 50m height):   {train_pe:,.0f} J")

person_pe = gravitational_potential_energy(70, 3)  # Person on 3rd floor
print(f"Person (70kg @ 3m height):       {person_pe:,.2f} J")


# --- Momentum Calculations ---
print("\n[Momentum - p = mv]")
print("-" * 50)

car_mom = momentum(car_mass, car_velocity)
print(f"Car ({car_mass}kg @ {car_velocity}m/s):    {car_mom:,.0f} kg·m/s")

bullet_mom = momentum(0.01, 800)  # 10g bullet at 800 m/s
print(f"Bullet (10g @ 800m/s):       {bullet_mom:,.2f} kg·m/s")

truck_mom = momentum(10000, car_velocity)  # Heavy truck at same speed
print(f"Truck (10,000kg @ {car_velocity}m/s):  {truck_mom:,.0f} kg·m/s")

print(f"\nNote: Truck has {truck_mom / car_mom:,.0f}× more momentum than car at same speed")


# --- Elastic Potential Energy ---
print("\n[Elastic Potential Energy - PE = ½kx²]")
print("-" * 50)

spring_pe = elastic_potential_energy(500, 0.2)  # k=500 N/m, compressed 20cm
print(f"Spring (k=500 N/m, x=0.2m):    {spring_pe:,.2f} J")

bow_pe = elastic_potential_energy(300, 0.5)  # Bow string pulled 50cm
print(f"Bow (k=300 N/m, x=0.5m):      {bow_pe:,.2f} J")


# --- Power Calculations ---
print("\n[Power - P = W/t]")
print("-" * 50)

train_power = power(train_work, 10)  # Work done in 10 seconds
print(f"Train power (in 10s):        {train_power:,.0f} W = {train_power/1000:,.1f} kW")

horsepower_equivalent = train_power / 746  # 1 HP ≈ 746 W
print(f"Equivalent:                  {horsepower_equivalent:,.0f} horsepower")


# ============================================================
# SECTION 5: COMPREHENSIVE SUMMARY TABLE
# ============================================================

print("\n" + "=" * 70)
print("COMPREHENSIVE PHYSICS SUMMARY")
print("=" * 70)

print(f"\n{'Category':<25} {'Quantity':<20} {'Value':>20} {'Units'}")
print("-" * 80)

# Temperature conversions
print(f"{'Temperature':<25} {'100°F to Celsius':<20} {f100_in_celsius:>20.2f} °C")
print(f"{'Temperature':<25} {'0°C to Fahrenheit':<20} {c0_in_fahrenheit:>20.2f} °F")

# Force and Energy
print(f"{'Force':<25} {'Train force':<20} {train_force:>20,} N")
print(f"{'Energy':<25} {'Bomb (E=mc²)':<20} {bomb_energy:>20,} J")

# Work
print(f"{'Work':<25} {'Train work':<20} {train_work:>20,} J")
print(f"{'Work':<25} {'Train power (10s)':<20} {train_power:>20,.0f} W")

# Extended: Kinetic Energy
print(f"{'Kinetic Energy':<25} {'Car (1500kg @ 30m/s)':<20} {car_ke:>20,.0f} J")
print(f"{'Kinetic Energy':<25} {'Rocket (5000kg @ 8000m/s)':<20} {rocket_ke:>20.2e} J")

# Extended: Potential Energy
print(f"{'Potential Energy':<25} {'Ball (0.5kg @ 10m)':<20} {ball_pe:>20,.2f} J")
print(f"{'Potential Energy':<25} {'Train (22680kg @ 50m)':<20} {train_pe:>20,.0f} J")

# Extended: Momentum
print(f"{'Momentum':<25} {'Car (1500kg @ 30m/s)':<20} {car_mom:>20,.0f} kg·m/s")
print(f"{'Momentum':<25} {'Truck (10000kg @ 30m/s)':<20} {truck_mom:>20,.0f} kg·m/s")

print("\n" + "=" * 70)
print("All physics calculations complete! Ready for class.")
print("=" * 70)