# api/middleware/auth.py
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from api.models import User

# Set auto_error=False to allow requests without auth header (for development)
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> User:
    """
    Extract and validate user from JWT token.
    For development: accepts any token or no token (returns mock user).
    For production: replace with actual Supabase/JWT validation.
    """
    # For development: if no auth provided, return mock user
    # TODO: Remove this in production and require valid authentication!
    if credentials is None:
        return User(id="dev_user", email="dev@example.com")
    
    token = credentials.credentials
    
    # TODO: Validate token with Supabase
    # For now, accept any token and return a mock user
    # In production, decode and validate the JWT token here:
    # from supabase import create_client, Client
    # supabase: Client = create_client(url, key)
    # user = supabase.auth.get_user(token)
    
    try:
        # Placeholder: Accept any token for development
        # Replace with actual validation in production
        return User(id="user_123", email="user@example.com")
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

