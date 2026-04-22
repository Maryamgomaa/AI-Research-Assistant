import logging
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from .models import SessionLocal, User
from .config import Config
import os

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")  # Set via env var in production!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
security = HTTPBearer()
security_optional = HTTPBearer(auto_error=False)
_log_auth = logging.getLogger("auth")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_optional),
    db: Session = Depends(get_db),
) -> Optional[User]:
    if not credentials:
        return None
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
    except JWTError:
        return None
    return db.query(User).filter(User.username == username).first()


def verify_llm_webhook_access(
    x_webhook_secret: Optional[str] = Header(None, alias="X-Webhook-Secret"),
    user: Optional[User] = Depends(get_current_user_optional),
) -> bool:
    """
    /webhook/research and /webhook/chatbot: require X-Webhook-Secret == WEBHOOK_LLM_SECRET
    OR a valid Bearer session (Web UI). Exception: ALLOW_ANONYMOUS_LLM_WEBHOOK for local dev only.
    """
    ws = (Config.WEBHOOK_LLM_SECRET or "").strip()
    if ws and (x_webhook_secret or "").strip() == ws:
        return True
    if user is not None:
        return True
    if Config.ALLOW_ANONYMOUS_LLM_WEBHOOK:
        _log_auth.warning(
            "[SECURITY] Unauthenticated LLM webhook allowed (ALLOW_ANONYMOUS_LLM_WEBHOOK=true)"
        )
        return True
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Provide X-Webhook-Secret (automation) or Authorization: Bearer (signed-in user)",
    )


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user