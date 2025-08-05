import hashlib
from datetime import datetime, timedelta
from bson import ObjectId
from database import db
from config import settings

class SessionManager:
    async def create_session(self, user_id: str, company_id: str) -> str:
        sid = hashlib.md5(f"{user_id}{datetime.utcnow()}".encode()).hexdigest()
        await db["chat_sessions"].insert_one({
            "session_id": sid,
            "user_id": ObjectId(user_id),
            "company": ObjectId(company_id),
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow()+timedelta(hours=settings.websocket_port)
        })
        return sid

    async def get(self, session_id: str):
        return await db["chat_sessions"].find_one({"session_id": session_id})

session_manager = SessionManager()
