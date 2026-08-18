import random
import string
import secrets
import math

AMBIGUOUS = set("0O1lI|`'\"")

def build_charset(use_upper=True, use_lower=True, use_digits=True, use_symbols=True, exclude_ambiguous=False):
    chars = ''
    if use_upper:
        chars += string.ascii_uppercase
    if use_lower:
        chars += string.ascii_lowercase
    if use_digits:
        chars += string.digits
    if use_symbols:
        chars += string.punctuation
    if exclude_ambiguous:
        chars = ''.join(c for c in chars if c not in AMBIGUOUS)
    return chars

def generate_password(length=12, use_upper=True, use_lower=True, use_digits=True,
                      use_symbols=True, ensure_each=True, exclude_ambiguous=False,
                      min_length=6, secure=False):
    """Generate a random password with optional constraints.
    secure=True uses the secrets module (cryptographically stronger).
    Returns (password_str, error_message_or_None)
    """
    if length < min_length:
        return None, f"Password too short! Choose at least {min_length} characters."
    
    pools = {}
    if use_upper:
        p = string.ascii_uppercase
        if exclude_ambiguous:
            p = ''.join(c for c in p if c not in AMBIGUOUS)
        if p:
            pools['upper'] = p
    if use_lower:
        p = string.ascii_lowercase
        if exclude_ambiguous:
            p = ''.join(c for c in p if c not in AMBIGUOUS)
        if p:
            pools['lower'] = p
    if use_digits:
        p = string.digits
        if exclude_ambiguous:
            p = ''.join(c for c in p if c not in AMBIGUOUS)
        if p:
            pools['digits'] = p
    if use_symbols:
        p = string.punctuation
        if exclude_ambiguous:
            p = ''.join(c for c in p if c not in AMBIGUOUS)
        if p:
            pools['symbols'] = p
    
    if not pools:
        return None, "No character types selected (or all excluded as ambiguous)."
    
    all_chars = ''.join(pools.values())
    chooser = secrets.choice if secure else random.choice
    
    def multi_choice(seq, k):
        if secure:
            return [secrets.choice(seq) for _ in range(k)]
        return random.choices(seq, k=k)
    
    password_chars = []
    
    if ensure_each and length >= len(pools):
        for pool in pools.values():
            password_chars.append(chooser(pool))
        remaining = length - len(password_chars)
        password_chars.extend(multi_choice(all_chars, remaining))
    else:
        password_chars = multi_choice(all_chars, length)
    
    # Shuffle
    if secure:
        for i in range(len(password_chars) - 1, 0, -1):
            j = secrets.randbelow(i + 1)
            password_chars[i], password_chars[j] = password_chars[j], password_chars[i]
    else:
        random.shuffle(password_chars)
    
    return ''.join(password_chars), None

def assess_strength(password):
    """Return (rating, score 0-10, details dict)"""
    if not password:
        return "Invalid", 0, {}
    length = len(password)
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_symbol = any(c in string.punctuation for c in password)
    diversity = sum([has_upper, has_lower, has_digit, has_symbol])
    
    score = 0
    if length >= 16:
        score += 4
    elif length >= 12:
        score += 3
    elif length >= 10:
        score += 2
    elif length >= 8:
        score += 1
    score += diversity  # 0-4
    if diversity >= 3 and length >= 10:
        score += 1
    if diversity == 4 and length >= 12:
        score += 1
    score = min(score, 10)
    
    if score >= 8:
        rating = "Strong"
    elif score >= 5:
        rating = "Medium"
    else:
        rating = "Weak"
    
    # Align with book length guideline
    if length < 6:
        rating = "Weak"
    elif length <= 10 and rating == "Strong":
        rating = "Medium"
    
    details = {
        'length': length,
        'has_upper': has_upper,
        'has_lower': has_lower,
        'has_digit': has_digit,
        'has_symbol': has_symbol,
        'diversity': diversity,
        'score': score
    }
    return rating, score, details

def estimate_entropy_bits(length, charset_size):
    if length <= 0 or charset_size <= 1:
        return 0.0
    return length * math.log2(charset_size)

if __name__ == "__main__":
    random.seed(42)
    print("=== Demo generate_password ===")
    for L in [4, 8, 12, 16]:
        pw, err = generate_password(L, ensure_each=True)
        if err:
            print(L, err)
        else:
            rating, score, det = assess_strength(pw)
            print(f"len={L}: {pw}  [{rating} score={score}]  div={det['diversity']}")
    print("\n=== Secure ===")
    pw, _ = generate_password(14, secure=True)
    print(pw, assess_strength(pw)[0])
