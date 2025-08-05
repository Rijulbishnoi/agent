import socketio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from bson import ObjectId
from datetime import datetime

from config import settings
from database import db
from session_manager import session_manager
from agents.mongodb_toolkit_agent import agent_executor

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                   allow_methods=["*"], allow_headers=["*"])

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins=settings.cors_origins)
socket_app = socketio.ASGIApp(sio, app)
sessions = {}

@sio.event
async def connect(sid, environ):
    await sio.emit("status", {"connected": True}, room=sid)

@sio.event
async def disconnect(sid):
    sessions.pop(sid, None)

@sio.event
async def authenticate(sid, data):
    uid = data.get("_id")
    if not uid or not ObjectId.is_valid(uid):
        return await sio.emit("auth_error", {"message":"Invalid _id"}, room=sid)
    user = await db["users"].find_one({"_id":ObjectId(uid)})
    if not user:
        return await sio.emit("auth_error", {"message":"User not found"}, room=sid)
    session_id = await session_manager.create_session(uid, str(user["company"]))
    sessions[sid] = {"_id":uid, "company":str(user["company"]), "session_id":session_id}
    await sio.emit("auth_success", {"session_id":session_id}, room=sid)

@sio.event
async def send_query(sid, data):
    ctx = sessions.get(sid)
    if not ctx:
        return await sio.emit("auth_error", {"message":"Not authenticated"}, room=sid)
    query = data.get("query","")
    # Invoke ReAct agent with dynamic MongoDB tools
    result = await agent_executor.ainvoke({"input": query})
    # If agent called a tool, it's already executed dynamically by toolkit.
    # The final `result["output"]` is the response.
    await sio.emit("query_response", {
        "response": result["output"],
        "timestamp": datetime.utcnow().isoformat()
    }, room=sid)

if __name__=="__main__":
    import uvicorn
    uvicorn.run(socket_app, host="0.0.0.0", port=settings.websocket_port)
