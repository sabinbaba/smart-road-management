"""
auth.py — Login / Signup / JWT Token logic
Smart Traffic Detection System
"""

from datetime import datetime, timedelta
from typing import Optional
import hashlib
import hmac
import base64
import json
import os

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
SECRET_KEY      = os.environ.get("SECRET_KEY", "smart-traffic-secret-key-2026")
TOKEN_EXPIRE_HOURS = 24

# ─────────────────────────────────────────────
#  PASSWORD HASHING
# ─────────────────────────────────────────────
def hash_password(password: str) -> str:
    """Hash a password using SHA-256 + salt."""
    salt = os.urandom(32)
    key  = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return base64.b64encode(salt + key).decode()

def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against its stored hash."""
    try:
        decoded = base64.b64decode(stored_hash.encode())
        salt    = decoded[:32]
        stored_key = decoded[32:]
        key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
        return hmac.compare_digest(key, stored_key)
    except Exception:
        return False

# ─────────────────────────────────────────────
#  JWT TOKEN (simple implementation)
# ─────────────────────────────────────────────
def _b64encode(data: dict) -> str:
    return base64.urlsafe_b64encode(
        json.dumps(data).encode()
    ).rstrip(b"=").decode()

def _b64decode(data: str) -> dict:
    padding = 4 - len(data) % 4
    data += "=" * padding
    return json.loads(base64.urlsafe_b64decode(data).decode())

def _sign(header: str, payload: str) -> str:
    msg = f"{header}.{payload}".encode()
    sig = hmac.new(SECRET_KEY.encode(), msg, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(sig).rstrip(b"=").decode()

def create_token(user_id: int, username: str, role: str) -> str:
    """Create a simple JWT-like token."""
    header  = _b64encode({"alg": "HS256", "typ": "JWT"})
    payload = _b64encode({
        "sub":      str(user_id),
        "username": username,
        "role":     role,
        "exp":      (datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)).isoformat()
    })
    signature = _sign(header, payload)
    return f"{header}.{payload}.{signature}"

def verify_token(token: str) -> Optional[dict]:
    """Verify token and return payload if valid."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None

        header, payload, signature = parts

        # Verify signature
        expected_sig = _sign(header, payload)
        if not hmac.compare_digest(signature, expected_sig):
            return None

        # Decode payload
        data = _b64decode(payload)

        # Check expiry
        exp = datetime.fromisoformat(data["exp"])
        if datetime.utcnow() > exp:
            return None

        return data
    except Exception:
        return None

# ─────────────────────────────────────────────
#  VALIDATION
# ─────────────────────────────────────────────
def validate_signup(username: str, email: str, password: str) -> list:
    """Return list of validation errors."""
    errors = []

    if not username or len(username) < 3:
        errors.append("Username must be at least 3 characters.")

    if not email or "@" not in email:
        errors.append("Invalid email address.")

    if not password or len(password) < 6:
        errors.append("Password must be at least 6 characters.")

    return errors

if __name__ == "__main__":
    # Quick test
    pwd    = "test1234"
    hashed = hash_password(pwd)
    print(f"Hash    : {hashed[:40]}...")
    print(f"Verify  : {verify_password(pwd, hashed)}")
    print(f"Wrong   : {verify_password('wrong', hashed)}")

    token = create_token(1, "admin", "admin")
    print(f"\nToken   : {token[:60]}...")
    data  = verify_token(token)
    print(f"Payload : {data}")
