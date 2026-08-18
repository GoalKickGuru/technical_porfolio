"""
Multi-Factor Authentication (MFA / 2FA) educational module
Pure Python stdlib implementation of TOTP (RFC 6238) + simple password hashing.
"""
import hmac
import hashlib
import struct
import time
import base64
import secrets
import json
from typing import Optional, Tuple, Dict, List

# ---------- Password helpers ----------
def hash_password(password: str, salt: bytes = None) -> Tuple[str, str]:
    """Return (salt_hex, hash_hex) using PBKDF2-HMAC-SHA256."""
    if salt is None:
        salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100_000)
    return salt.hex(), dk.hex()

def verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    salt = bytes.fromhex(salt_hex)
    dk = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100_000)
    return hmac.compare_digest(dk.hex(), hash_hex)

# ---------- TOTP (RFC 6238) ----------
def generate_totp_secret(nbytes: int = 20) -> str:
    """Generate a base32-encoded secret suitable for authenticator apps."""
    raw = secrets.token_bytes(nbytes)
    return base64.b32encode(raw).decode('ascii').rstrip('=')

def _b32decode(secret: str) -> bytes:
    secret = secret.upper().replace(' ', '')
    pad = '=' * ((8 - len(secret) % 8) % 8)
    return base64.b32decode(secret + pad)

def hotp(secret: str, counter: int, digits: int = 6) -> str:
    """HMAC-based One-Time Password."""
    key = _b32decode(secret)
    msg = struct.pack('>Q', counter)
    digest = hmac.new(key, msg, hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    code_int = struct.unpack('>I', digest[offset:offset + 4])[0] & 0x7FFFFFFF
    return str(code_int % (10 ** digits)).zfill(digits)

def totp(secret: str, digits: int = 6, interval: int = 30, for_time: float = None) -> str:
    """Time-based OTP. for_time allows testing with a fixed timestamp."""
    if for_time is None:
        for_time = time.time()
    counter = int(for_time // interval)
    return hotp(secret, counter, digits)

def verify_totp(secret: str, code: str, window: int = 1, digits: int = 6,
                interval: int = 30, for_time: float = None) -> bool:
    """Verify a TOTP code, allowing ±window time steps."""
    if for_time is None:
        for_time = time.time()
    code = code.strip().zfill(digits)
    for delta in range(-window, window + 1):
        expected = totp(secret, digits, interval, for_time + delta * interval)
        if hmac.compare_digest(expected, code):
            return True
    return False

# ---------- Backup codes ----------
def generate_backup_codes(n: int = 8, length: int = 8) -> List[str]:
    """Generate one-time backup codes (alphanumeric)."""
    alphabet = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'  # no ambiguous
    return [''.join(secrets.choice(alphabet) for _ in range(length)) for _ in range(n)]

# ---------- Simple in-memory user store ----------
class UserStore:
    def __init__(self):
        self.users: Dict[str, dict] = {}

    def register(self, username: str, password: str, enable_mfa: bool = True) -> dict:
        if username in self.users:
            raise ValueError("Username already exists")
        salt, pw_hash = hash_password(password)
        secret = generate_totp_secret() if enable_mfa else None
        backups = generate_backup_codes() if enable_mfa else []
        self.users[username] = {
            'salt': salt,
            'hash': pw_hash,
            'totp_secret': secret,
            'mfa_enabled': enable_mfa,
            'backup_codes': backups,
            'failed_attempts': 0,
            'locked_until': 0.0,
        }
        return {
            'username': username,
            'totp_secret': secret,
            'backup_codes': backups.copy(),
            'mfa_enabled': enable_mfa,
        }

    def authenticate(self, username: str, password: str, otp: str = None,
                     use_backup: bool = False, max_attempts: int = 5,
                     lockout_seconds: int = 300) -> Tuple[bool, str]:
        """Full MFA login. Returns (success, message)."""
        user = self.users.get(username)
        if not user:
            return False, "Unknown user"
        now = time.time()
        if user['locked_until'] > now:
            remaining = int(user['locked_until'] - now)
            return False, f"Account locked. Try again in {remaining}s"
        if not verify_password(password, user['salt'], user['hash']):
            user['failed_attempts'] += 1
            if user['failed_attempts'] >= max_attempts:
                user['locked_until'] = now + lockout_seconds
                user['failed_attempts'] = 0
                return False, "Too many failures – account locked"
            return False, "Invalid password"
        # Password OK
        if not user['mfa_enabled']:
            user['failed_attempts'] = 0
            return True, "Login successful (MFA disabled)"
        # MFA required
        if use_backup and otp:
            if otp.upper() in user['backup_codes']:
                user['backup_codes'].remove(otp.upper())
                user['failed_attempts'] = 0
                return True, "Login successful (backup code used)"
            user['failed_attempts'] += 1
            return False, "Invalid backup code"
        if otp is None or not verify_totp(user['totp_secret'], otp):
            user['failed_attempts'] += 1
            if user['failed_attempts'] >= max_attempts:
                user['locked_until'] = now + lockout_seconds
                user['failed_attempts'] = 0
                return False, "Too many failures – account locked"
            return False, "Invalid OTP"
        user['failed_attempts'] = 0
        return True, "Login successful (MFA verified)"

# ---------- Demo / self-test ----------
if __name__ == "__main__":
    store = UserStore()
    info = store.register("alice", "S3cureP@ss!", enable_mfa=True)
    print("Registered alice")
    print("TOTP secret:", info['totp_secret'])
    print("Backup codes:", info['backup_codes'])
    current = totp(info['totp_secret'])
    print("Current TOTP :", current)
    ok, msg = store.authenticate("alice", "S3cureP@ss!", current)
    print("Auth result  :", ok, msg)
    ok2, msg2 = store.authenticate("alice", "wrong", current)
    print("Bad pwd      :", ok2, msg2)
    # backup
    bc = info['backup_codes'][0]
    ok3, msg3 = store.authenticate("alice", "S3cureP@ss!", bc, use_backup=True)
    print("Backup login :", ok3, msg3)
