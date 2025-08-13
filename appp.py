import time
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from bson import ObjectId
from agents.mongodb_toolkit_agenp import process_chatbot_query, user_context_manager

# Initialize FastAPI app
app = FastAPI(
    title="MongoDB Chatbot API",
    description="AI-powered chatbot with MongoDB integration",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class QueryRequest(BaseModel):
    user_id: str = Field(..., description="User ID for company-based filtering")
    query: str = Field(..., min_length=1, max_length=1000, description="User query message")
    
    @validator('user_id')
    def validate_user_id(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError('Invalid user_id format')
        return v
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()

class QueryResponse(BaseModel):
    status: str
    response: Optional[str] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str

# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.post("/query", response_model=QueryResponse)
async def query_bot(request: QueryRequest):
    """Process chatbot query with MongoDB integration"""
    start_time = time.time()
    
    try:
        # Pre-validate user exists and has company
        try:
            user_context = await user_context_manager.get_user_context(request.user_id)
            print(f"🔍 Pre-validation - User: {request.user_id}, Company: {user_context.get('company_id')}")
        except ValueError as e:
            return QueryResponse(
                status="error",
                error=f"User validation failed: {str(e)}",
                execution_time=time.time() - start_time
            )
        
        # Use the optimized process_chatbot_query function
        result = await process_chatbot_query(request.query, request.user_id)
        
        execution_time = time.time() - start_time
        
        if result["status"] == "success":
            return QueryResponse(
                status="success",
                response=result["response"],
                execution_time=execution_time
            )
        else:
            return QueryResponse(
                status="error",
                error=result["message"],
                execution_time=execution_time
            )
            
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"❌ Unexpected error in query_bot: {str(e)}")  # Debug log
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error", 
                "error": f"Internal server error: {str(e)}",
                "execution_time": execution_time
            }
        )

@app.post("/query/batch")
async def query_bot_batch(requests: list[QueryRequest]):
    """Process multiple queries in batch (max 5 per request)"""
    if len(requests) > 5:
        raise HTTPException(
            status_code=400,
            detail="Maximum 5 queries allowed per batch request"
        )
    
    results = []
    start_time = time.time()
    
    for req in requests:
        try:
            result = await process_chatbot_query(req.query, req.user_id)
            results.append({
                "user_id": req.user_id,
                "query": req.query,
                "result": result
            })
        except Exception as e:
            results.append({
                "user_id": req.user_id,
                "query": req.query,
                "result": {"status": "error", "message": str(e)}
            })
    
    return {
        "status": "completed",
        "results": results,
        "total_queries": len(requests),
        "execution_time": time.time() - start_time
    }

@app.get("/user/{user_id}/info")
async def get_user_info(user_id: str):
    """Get user information and context"""
    try:
        if not ObjectId.is_valid(user_id):
            raise HTTPException(status_code=400, detail="Invalid user_id format")
        
        user_context = await user_context_manager.get_user_context(user_id)
        
        return {
            "status": "success",
            "user_id": user_id,
            "user_name": user_context["user_name"],
            "user_email": user_context["user_email"],
            "company_id": str(user_context["company_id"]) if user_context["company_id"] else None
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@app.delete("/cache/user/{user_id}")
async def clear_user_cache(user_id: str):
    """Clear cache for specific user"""
    try:
        if not ObjectId.is_valid(user_id):
            raise HTTPException(status_code=400, detail="Invalid user_id format")
        
        user_context_manager.clear_cache(user_id)
        
        return {
            "status": "success",
            "message": f"Cache cleared for user {user_id}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Cache clear failed")

@app.delete("/cache/all")
async def clear_all_cache():
    """Clear all user cache"""
    try:
        user_context_manager.clear_cache()
        return {
            "status": "success",
            "message": "All cache cleared"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Cache clear failed")

# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={
            "status": "error",
            "error": str(exc),
            "detail": "Invalid input provided"
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error": "Internal server error",
            "detail": "An unexpected error occurred"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000
    )
