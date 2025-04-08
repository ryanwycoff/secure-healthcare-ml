# secure-healthcare-ml/api/auth.py

"""
Authentication module for the Secure Healthcare ML service.
This module handles user authentication and authorization,
using JWT tokens for secure API access.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from typing import List
from datetime import datetime, timedelta
from .schemas import User, TokenData
from .utils import verify_password, get_user_by_username
from .config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES

# OAuth2 password bearer for retrieving the token from Authorization header
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Utility function to create a JWT token
def create_access_token(data: dict, expires_delta: timedelta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Utility function to decode the JWT token
def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return TokenData(username=username)
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

# Dependency to get the current user from the token
def get_current_user(token: str = Depends(oauth2_scheme)):
    return decode_access_token(token)

# Dependency to check if the user has admin privileges
def get_current_admin_user(current_user: User = Depends(get_current_user)):
    if not current_user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient privileges")
    return current_user

# Function to authenticate a user using their credentials
def authenticate_user(username: str, password: str):
    user = get_user_by_username(username)
    if user is None or not verify_password(password, user.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    return user
