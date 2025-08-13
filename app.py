# fastapi_chatbot_api.py

from fastapi import FastAPI
from pydantic import BaseModel
from agents.mongodb_toolkit_agena import agent_executor

app = FastAPI()

class QueryRequest(BaseModel):
    user_id: str
    query: str

@app.post("/query")
async def query_bot(request: QueryRequest):
    try:
        # Pass user_id to the agent for permission-based filtering
        response = await agent_executor.ainvoke({
            "input": request.query,
            "user_id": request.user_id
        })
        return {"response": response["output"]}
    except Exception as err:
        return {"error": str(err)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)