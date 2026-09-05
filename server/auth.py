"""
Phase 7: API key authentication.

Simple but real: every request to a protected endpoint must include
a header:  Authorization: Bearer <api_key>

For a portfolio project, keys live in a Python dict (in-memory).
In a real production system, this would be a database table with
hashed keys, usage tracking, and expiry - but the CONCEPT (verify
identity before doing expensive work) is the same either way.
"""

from fastapi import Header, HTTPException, status
from typing import Optional

# In-memory "database" of valid API keys.
# key -> metadata about who owns it (useful for logging/rate limiting later)
VALID_API_KEYS = {
    "demo-key-12345": {"owner": "demo-user", "tier": "free"},
    "vignesh-dev-key": {"owner": "vignesh", "tier": "unlimited"},
}


def verify_api_key(authorization: Optional[str] = Header(None)) -> dict:
    """
    FastAPI dependency: runs automatically before the endpoint function,
    on every request that declares it as a dependency (see server/main.py).

    Expects header format:  Authorization: Bearer <api_key>
    Returns the key's metadata dict if valid.
    Raises 401 Unauthorized if missing/invalid - the endpoint function
    never even runs in that case, saving compute on bad requests.
    """
    if authorization is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header. Expected: Bearer <api_key>"
        )

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Expected: Bearer <api_key>"
        )

    api_key = parts[1]

    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    return {"api_key": api_key, **VALID_API_KEYS[api_key]}
