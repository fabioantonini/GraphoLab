"""
GraphoLab Backend — Password hashing with bcrypt.
Uses bcrypt directly (passlib 1.7 is incompatible with bcrypt 4.x).
"""

import bcrypt


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())
