import math
import random
import string

# Optional dependency for clipboard support
try:
    import pyperclip
    HAS_PYPERCLIP = True
except ImportError:
    HAS_PYPERCLIP = False


def calculate_entropy(password: str) -> float:
    """Calculates entropy in bits based on character pool size and length."""
    pool_size = 0
    if any(c in string.ascii_lowercase for c in password):
        pool_size += 26
    if any(c in string.ascii_uppercase for c in password):
        pool_size += 26
    if any(c in string.digits for c in password):
        pool_size += 10
    if any(c in string.punctuation for c in password):
        pool_size += len(string.punctuation)

    if pool_size == 0 or len(password) == 0:
        return 0.0

    return len(password) * math.log2(pool_size)


def evaluate_strength(password: str) -> str:
    """Evaluates password strength based on entropy and length."""
    entropy = calculate_entropy(password)
    length = len(password)

    if length < 6 or entropy < 28:
        return "Weak 🔴"
    elif length < 10 or entropy < 50:
        return "Medium 🟡"
    elif entropy < 80:
        return "Strong 🟢"
    else:
        return "Very Strong 🔥"


def generate_password(
    length: int = 12,
    use_upper: bool = True,
    use_lower: bool = True,
    use_digits: bool = True,
    use_symbols: bool = True,
    guarantee_each: bool = True,
) -> str:
    """
    Generates a secure, random password with custom options and guaranteed character types.
    """
    pools = []
    if use_lower:
        pools.append(string.ascii_lowercase)
    if use_upper:
        pools.append(string.ascii_uppercase)
    if use_digits:
        pools.append(string.digits)
    if use_symbols:
        pools.append(string.punctuation)

    if not pools:
        raise ValueError("At least one character set must be selected.")

    if length < len(pools) and guarantee_each:
        raise ValueError(
            f"Length must be at least {len(pools)} to include all selected character types."
        )

    password_chars = []

    # Ensure at least one character from each chosen set is included
    if guarantee_each:
        for pool in pools:
            password_chars.append(random.choice(pool))

    # Fill the remainder of the password length from the combined pool
    all_characters = "".join(pools)
    remaining_length = length - len(password_chars)
    password_chars.extend(random.choices(all_characters, k=remaining_length))

    # Shuffle to ensure guaranteed characters aren't predictable at the start
    random.shuffle(password_chars)

    return "".join(password_chars)


def get_bool_input(prompt: str, default: bool = True) -> bool:
    """Helper to get y/n prompt response with a default value."""
    default_str = "[Y/n]" if default else "[y/N]"
    user_input = input(f"{prompt} {default_str}: ").strip().lower()
    if not user_input:
        return default
    return user_input.startswith("y")


def main():
    print("=" * 45)
    print("   🔐 ENHANCED SECURE PASSWORD GENERATOR 🔐   ")
    print("=" * 45)

    # Get password length safely
    while True:
        try:
            length = int(input("Enter desired password length (min 6): "))
            if length < 6:
                print("⚠️  Security Warning: Length should be at least 6.")
                if not get_bool_input("Proceed anyway?", default=False):
                    continue
            break
        except ValueError:
            print("❌ Invalid input! Please enter a valid integer.")

    # Character set options
    print("\n--- Character Set Options ---")
    use_lower = get_bool_input("Include Lowercase letters (a-z)?")
    use_upper = get_bool_input("Include Uppercase letters (A-Z)?")
    use_digits = get_bool_input("Include Digits (0-9)?")
    use_symbols = get_bool_input("Include Symbols (!@#$%...)?")

    try:
        password = generate_password(
            length=length,
            use_lower=use_lower,
            use_upper=use_upper,
            use_digits=use_digits,
            use_symbols=use_symbols,
            guarantee_each=True,
        )
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        return

    strength = evaluate_strength(password)
    entropy = calculate_entropy(password)

    print("\n" + "=" * 45)
    print(f"🔑 Generated Password : {password}")
    print(f"📊 Strength Rating    : {strength}")
    print(f"🧮 Estimated Entropy  : {entropy:.1f} bits")
    print("=" * 45)

    # Clipboard copy option
    copy = get_bool_input("\nCopy password to clipboard?")
    if copy:
        if HAS_PYPERCLIP:
            pyperclip.copy(password)
            print("✅ Password copied to clipboard!")
        else:
            print(
                "⚠️  'pyperclip' module is not installed. Install it via 'pip install pyperclip' to use clipboard feature."
            )


if __name__ == "__main__":
    main()