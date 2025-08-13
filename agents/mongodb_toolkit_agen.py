import os
import json
import re
import time
import threading
import asyncio
import logging
from datetime import datetime, timedelta
from contextlib import contextmanager
from functools import lru_cache
from typing import Optional, Type, Any, Dict

import google.generativeai as genai
from config import settings
from database import db
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from bson import ObjectId


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Google Gemini API key
genai.configure(api_key=settings.gemini_api_key)

# Initialize Gemini LLM wrapper
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=settings.gemini_api_key,
    temperature=0.0
)


def convert_to_objectids(query_dict: Dict[str, Any], objectid_fields: list) -> Dict[str, Any]:
    """
    Recursively convert string values to ObjectIds for specified fields with validation
    """
    result = {}
    for key, value in query_dict.items():
        if key in objectid_fields and isinstance(value, str):
            if not ObjectId.is_valid(value):
                raise ValueError(f"Invalid ObjectId format for field '{key}': {value}")
            result[key] = ObjectId(value)
        elif isinstance(value, dict):
            result[key] = convert_to_objectids(value, objectid_fields)
        elif isinstance(value, list):
            result[key] = [
                convert_to_objectids(item, objectid_fields) if isinstance(item, dict) 
                else (ObjectId(item) if key in objectid_fields and isinstance(item, str) and ObjectId.is_valid(item) else item)
                for item in value
            ]
        else:
            result[key] = value
    return result


def replace_placeholders_recursively(obj, user_id, user_company):
    """Recursively replace placeholders in any nested structure"""
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            if value == "CURRENT_USER_ID":
                result[key] = ObjectId(user_id) if user_id else value
            elif value == "USER_COMPANY_ID":
                result[key] = ObjectId(user_company) if user_company else value
            else:
                result[key] = replace_placeholders_recursively(value, user_id, user_company)
        return result
    elif isinstance(obj, list):
        return [replace_placeholders_recursively(item, user_id, user_company) for item in obj]
    else:
        if obj == "CURRENT_USER_ID":
            return ObjectId(user_id) if user_id else obj
        elif obj == "USER_COMPANY_ID":
            return ObjectId(user_company) if user_company else obj
        else:
            return obj


def validate_user_input(user_input: str) -> bool:
    """Validate user input for potential injection attacks"""
    dangerous_patterns = [
        r'\$where', r'javascript:', r'eval\s*\(', r'function\s*\(',
        r'this\.', r'db\.', r'load\s*\(', r'sleep\s*\('
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, user_input, re.IGNORECASE):
            return False
    return True


# Define ObjectId fields for each collection
OBJECTID_FIELDS = {
    "leads": ["_id", "bhk", "bhkType", "broker", "company", "project"],
    "lead-assignments": ["_id", "assignee", "company", "defaultPrimary", "defaultSecondary", "lead", "team"],
    "lead-notes": ["_id", "company", "lead", "tag"],
    "lead-rotations": ["_id", "assignee", "company", "lead", "team"],
    "lead-visited-properties": ["_id", "company", "lead", "property"],
    "users": ["_id", "company", "designation", "groups[]"],
    "companies": ["_id", "country", "plan", "superAdmin"],
    "properties": ["_id", "bhk", "bhkType", "company", "project", "propertyUnitSubType"],
    "projects": ["_id", "company", "land", "category", "country"],
    "brokers": ["_id", "company", "country"],
    "tenants": ["_id", "company", "project", "property"],
    "rent-payments": ["_id", "company", "project", "property", "tenant"]
}


# Database Schema Knowledge
SCHEMA_KNOWLEDGE = """
Relevant Collections and Fields:
leads: ["_id", "bhk", "bhkType", "broker", "buyingTimeline", "commissionPercent", "company", "createdAt", "email", "leadStatus", "maxBudget", "minBudget", "name", "phone", "project", "propertyType", "rotationCount", "secondaryPhone", "sourceType", "status", "updatedAt"]
lead-assignments: ["_id", "assignee", "company", "createdAt", "defaultPrimary", "defaultSecondary", "lead", "status", "team", "updatedAt"]
lead-notes: ["_id", "company", "lead", "communicationType", "meetingDateTime", "nextSiteVisitScheduledDate", "remarks", "siteVisitScheduledDate", "siteVisitStatus", "status", "tag", "updatedAt"]
lead-rotations: ["_id", "assignee", "company", "createdAt", "date", "lead", "team", "updatedAt"]
lead-visited-properties: ["_id", "company", "createdAt", "lead", "property", "remarks", "status", "updatedAt"]
users: ["_id", "company", "designation", "email", "name", "phone", "groups[]", "status", "updatedAt"]

Relationships:
- lead-assignments: lead -> leads._id, assignee -> users._id, team -> teams._id
- lead-notes: lead -> leads._id, tag -> tags._id
- lead-rotations: lead -> leads._id, assignee -> users._id, team -> teams._id
- lead-visited-properties: lead -> leads._id, property -> properties._id

User Filtering:
- Only allow access to leads assigned to the current user (assignedTo/assignee).
- For users, only allow access to own user document unless super admin.
- Permission checks are enforced for each collection.

Field Value Clarifications:
- 'leadStatus' tracks stages such as 'New', 'In Progress', 'Closed - Won', 'Closed - Lost'.
- 'status' field indicates general activity, e.g., 'active', 'inactive'.

Example Queries:
- "Show my assigned leads": Query lead-assignments where assignee = CURRENT_USER_ID, lookup leads.
- "Show notes for a lead": Query lead-notes where lead = lead_id.
- "Show properties visited by a lead": Query lead-visited-properties where lead = lead_id.
- "Show my lead rotations": Query lead-rotations where assignee = CURRENT_USER_ID.
"""


class UserContextManager:
    def __init__(self):
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
        self._lock = asyncio.Lock()
    
    async def get_user_context(self, user_id: str) -> dict:
        async with self._lock:
            cache_key = f"user_{user_id}"
            current_time = datetime.now()
            
            # Check cache first
            if cache_key in self._cache:
                cached_data, timestamp = self._cache[cache_key]
                if current_time - timestamp < timedelta(seconds=self._cache_ttl):
                    return cached_data
            
            # Fetch from database
            try:
                user = await db.users.find_one({"_id": ObjectId(user_id)})
                if not user:
                    raise ValueError(f"User {user_id} not found")
                
                context = {
                    "user_id": user_id,
                    "company_id": user.get("company"),
                    "account_type": user.get("account_type", ""),
                    "permissions": user.get("permissions", {}),
                    "is_super_admin": user.get("account_type") == "company_super_admin",
                    "user_name": user.get("name"),
                    "user_email": user.get("email")
                }
                
                # Cache the result
                self._cache[cache_key] = (context, current_time)
                return context
                
            except Exception as e:
                raise ValueError(f"Failed to fetch user context: {str(e)}")
    
    def clear_cache(self, user_id: str = None):
        if user_id:
            cache_key = f"user_{user_id}"
            self._cache.pop(cache_key, None)
        else:
            self._cache.clear()


class QueryLogger:
    @staticmethod
    def log_query(user_id: str, collection: str, query_type: str, 
                  execution_time: float, result_count: int):
        logger.info(f"Query executed - User: {user_id}, Collection: {collection}, "
                   f"Type: {query_type}, Time: {execution_time:.2f}s, Results: {result_count}")
    
    @staticmethod
    def log_error(user_id: str, error: str, query: str):
        logger.error(f"Query error - User: {user_id}, Error: {error}, Query: {query[:100]}...")


# Global context manager
user_context_manager = UserContextManager()


class BaseMongoDBTool(BaseTool):
    async def get_user_context(self, user_id: str = None) -> dict:
        if not user_id:
            raise ValueError("User ID is required")
        return await user_context_manager.get_user_context(user_id)
    
    def validate_permissions(self, user_context: dict, collection: str, operation: str = "read") -> bool:
        if user_context["is_super_admin"]:
            return True
        
        collection_perms = user_context["permissions"].get(collection, {})
        return operation in collection_perms
    
    def apply_company_filter(self, query: dict, user_context: dict, collection: str) -> dict:
        if user_context["is_super_admin"]:
            return query
        
        company_id = user_context["company_id"]
        if not company_id:
            return query
        
        # Special handling for different collections
        if collection == "users":
            return {"_id": ObjectId(user_context["user_id"])}
        elif collection in ["lead-assignments", "lead-rotations"]:
            if "assignee" not in query:
                query["assignee"] = ObjectId(user_context["user_id"])
        elif "company" not in query:
            query["company"] = ObjectId(company_id)
        
        return query
    
    def _error_response(self, error_code: str, message: str) -> dict:
        return {
            "status": "error",
            "error_code": error_code,
            "message": message
        }


# Pydantic Models
class MongoDBQueryInput(BaseModel):
    query: str = Field(description="The MongoDB query in JSON format")
    user_id: Optional[str] = Field(description="User ID for permission-based filtering", default=None)
    collection: Optional[str] = Field(description="Collection name", default=None)
    count: Optional[bool] = Field(description="Whether to count documents", default=False)
    projection: Optional[str] = Field(description="Projection fields", default=None)
    limit: Optional[int] = Field(description="Limit number of results", default=None)


class EnhancedMongoDBQueryTool(BaseMongoDBTool):
    name: str = "mongodb_query"
    description: str = "Execute MongoDB queries with optimized permission checking and caching"
    args_schema: Type[BaseModel] = MongoDBQueryInput
    
    def _run(self, *args, **kwargs) -> str:
        return "This tool only supports async operations"
    
    async def _arun(
        self,
        query: str,
        user_id: Optional[str] = None,
        collection: Optional[str] = None,
        count: Optional[bool] = False,
        projection: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> dict:
        start_time = time.time()
        try:
            # Parse and validate inputs
            params = await self._parse_query_params(query, collection, count, projection, limit, user_id)
            
            # Get user context with caching
            user_context = await self.get_user_context(params["user_id"])
            
            # Validate collection exists
            if not await self._validate_collection(params["collection"]):
                return self._error_response("InvalidCollection", f"Collection '{params['collection']}' does not exist")
            
            # Check permissions
            if not self.validate_permissions(user_context, params["collection"], "read"):
                return self._error_response("PermissionDenied", f"No read permission for '{params['collection']}'")
            
            # Apply filters and execute query
            result = await self._execute_query(params, user_context)
            
            # Log successful query
            QueryLogger.log_query(
                params["user_id"], 
                params["collection"], 
                "query", 
                time.time() - start_time,
                result.get("count", 0)
            )
            
            return result
            
        except ValueError as e:
            QueryLogger.log_error(user_id or "unknown", str(e), query)
            return self._error_response("ValidationError", str(e))
        except Exception as e:
            QueryLogger.log_error(user_id or "unknown", str(e), query)
            return self._error_response("ExecutionError", f"Query execution failed: {str(e)}")
    
    async def _parse_query_params(self, query, collection, count, projection, limit, user_id) -> dict:
        params = {
            "collection": collection,
            "query": {},
            "count": count,
            "projection": None,
            "limit": limit,
            "user_id": user_id
        }
        
        if isinstance(query, str):
            try:
                query_data = json.loads(query)
                params.update({
                    "collection": query_data.get("collection") or collection,
                    "query": query_data.get("query", {}),
                    "count": query_data.get("count", count),
                    "limit": query_data.get("limit", limit),
                    "user_id": query_data.get("user_id", user_id)
                })
                
                proj_raw = query_data.get("projection") or projection
                if proj_raw:
                    params["projection"] = json.loads(proj_raw) if isinstance(proj_raw, str) else proj_raw
                    
            except json.JSONDecodeError:
                if not collection:
                    params["collection"] = query
        
        if not params["collection"]:
            raise ValueError("Collection name is required")
        if not params["user_id"]:
            raise ValueError("User ID is required")
            
        return params
    
    async def _validate_collection(self, collection_name: str) -> bool:
        collections = await db.list_collection_names()
        return collection_name in collections
    
    async def _execute_query(self, params: dict, user_context: dict) -> dict:
        # Process query with placeholders and ObjectIds
        mongo_query = self._process_query(params["query"], user_context, params["collection"])
        
        # Apply company/user filtering
        mongo_query = self.apply_company_filter(mongo_query, user_context, params["collection"])
        
        # Execute query
        coll = db[params["collection"]]
        
        if params["count"]:
            count = await coll.count_documents(mongo_query)
            return {"status": "success", "count": count}
        
        # Apply default projection if needed
        projection = params["projection"] or self._get_default_projection(params["collection"])
        
        cursor = coll.find(mongo_query, projection)
        if params["limit"]:
            cursor = cursor.limit(min(params["limit"], 100))  # Cap at 100
        
        documents = await cursor.to_list(length=params["limit"] or 50)
        
        return {
            "status": "success",
            "count": len(documents),
            "data": documents,
            "message": f"Found {len(documents)} {params['collection']}" if documents else "No documents found"
        }
    
    def _process_query(self, query: dict, user_context: dict, collection: str) -> dict:
        # Replace placeholders
        query = replace_placeholders_recursively(
            query, 
            user_context["user_id"], 
            user_context["company_id"]
        )
        
        # Convert ObjectIds
        if collection in OBJECTID_FIELDS:
            query = convert_to_objectids(query, OBJECTID_FIELDS[collection])
        
        return query
    
    def _get_default_projection(self, collection: str) -> dict:
        projections = {
            "leads": {"name": 1, "email": 1, "phone": 1, "leadStatus": 1, "_id": 1},
            "users": {"name": 1, "email": 1, "_id": 1},
            "companies": {"name": 1, "clientName": 1, "_id": 1},
            "properties": {"name": 1, "_id": 1},
            "lead-assignments": {"lead": 1, "assignee": 1, "status": 1, "createdAt": 1}
        }
        return projections.get(collection, {"name": 1, "_id": 1})


class MongoDBAggregationInput(BaseModel):
    pipeline: str = Field(description="The MongoDB aggregation pipeline in JSON format")
    user_id: Optional[str] = Field(description="User ID for permission-based filtering", default=None)
    collection: Optional[str] = Field(description="Collection name", default=None)
    stages: Optional[str] = Field(description="Aggregation stages", default=None)


class EnhancedMongoDBAggregationTool(BaseMongoDBTool):
    name: str = "mongodb_aggregation"
    description: str = "Execute MongoDB aggregation pipelines for complex queries with optimized performance"
    args_schema: Type[BaseModel] = MongoDBAggregationInput

    def _run(self, pipeline: str, user_id: Optional[str] = None) -> str:
        return "This tool only supports async operations"

    async def _arun(self, pipeline: str, user_id: Optional[str] = None, 
                   collection: Optional[str] = None, stages: Optional[str] = None) -> str:
        start_time = time.time()
        try:
            # Parse pipeline parameters
            collection_name, parsed_stages, parsed_user_id = await self._parse_pipeline(
                pipeline, collection, stages, user_id)
            
            # Get user context
            user_context = await self.get_user_context(parsed_user_id)
            
            # Check permissions
            if not self.validate_permissions(user_context, collection_name, "read"):
                return f"Error: No read permission for '{collection_name}'"
            
            # Process and execute aggregation
            result = await self._execute_aggregation(collection_name, parsed_stages, user_context)
            
            # Log successful aggregation
            result_count = len(result) if isinstance(result, list) else 1
            QueryLogger.log_query(
                parsed_user_id, collection_name, "aggregation", 
                time.time() - start_time, result_count)
            
            return self._format_aggregation_result(result, collection_name)
            
        except Exception as e:
            QueryLogger.log_error(user_id or "unknown", str(e), pipeline)
            return f"Error executing aggregation: {str(e)}"
    
    async def _parse_pipeline(self, pipeline, collection, stages, user_id):
        if isinstance(pipeline, str):
            try:
                pipeline_data = json.loads(pipeline)
                collection_name = pipeline_data.get("collection") or collection
                parsed_stages = pipeline_data.get("pipeline", []) or pipeline_data.get("stages", [])
                parsed_user_id = pipeline_data.get("user_id") or user_id
            except json.JSONDecodeError:
                collection_name = collection or pipeline
                parsed_stages = []
                parsed_user_id = user_id
        else:
            collection_name = collection
            parsed_stages = json.loads(stages) if isinstance(stages, str) else (stages or [])
            parsed_user_id = user_id
        
        if not collection_name:
            raise ValueError("Collection name is required")
        if not parsed_user_id:
            raise ValueError("User ID is required")
            
        return collection_name, parsed_stages, parsed_user_id
    
    async def _execute_aggregation(self, collection_name: str, stages: list, user_context: dict) -> list:
        # Replace placeholders in stages
        stages = replace_placeholders_recursively(
            stages, user_context["user_id"], user_context["company_id"])
        
        # Apply security filters
        stages = self._apply_security_filters(stages, user_context, collection_name)
        
        # Convert ObjectIds
        if collection_name in OBJECTID_FIELDS:
            for i, stage in enumerate(stages):
                if isinstance(stage, dict):
                    stages[i] = convert_to_objectids(stage, OBJECTID_FIELDS[collection_name])
        
        # Execute aggregation
        collection = db[collection_name]
        cursor = collection.aggregate(stages)
        return await cursor.to_list(length=100)
    
    def _apply_security_filters(self, stages: list, user_context: dict, collection_name: str) -> list:
        if user_context["is_super_admin"]:
            return stages
        
        # Check if user filter already exists
        has_user_filter = any(
            isinstance(stage, dict) and "$match" in stage and 
            any(field in stage.get("$match", {}) for field in ["_id", "assignedTo", "assignee", "company"])
            for stage in stages
        )
        
        if not has_user_filter:
            if collection_name == "users":
                user_filter = {"$match": {"_id": ObjectId(user_context["user_id"])}}
            elif collection_name in ["leads", "lead-assignments", "lead-rotations"]:
                user_filter = {"$match": {"assignee": ObjectId(user_context["user_id"])}}
            else:
                company_id = user_context["company_id"]
                if company_id:
                    user_filter = {"$match": {"company": ObjectId(company_id)}}
                else:
                    user_filter = {"$match": {}}
            stages.insert(0, user_filter)
        
        return stages
    
    def _format_aggregation_result(self, results: list, collection_name: str) -> str:
        if not results:
            return f"No results found for aggregation on {collection_name}."
        
        if len(results) == 1:
            return f"Aggregation result: {json.dumps(results[0], default=str, indent=2)}"
        else:
            return f"Aggregation results ({len(results)} items): {json.dumps(results, default=str, indent=2)}"


class MongoDBGetUserPermissionsInput(BaseModel):
    user_id: str = Field(description="The user ID to get permissions for")


class MongoDBGetUserPermissionsTool(BaseMongoDBTool):
    name: str = "mongodb_get_user_permissions"
    description: str = "Get user permissions for filtering queries"
    args_schema: Type[BaseModel] = MongoDBGetUserPermissionsInput

    def _run(self, user_id: str) -> str:
        return "This tool only supports async operations"

    async def _arun(self, user_id: Optional[str] = None) -> str:
        try:
            if not user_id:
                raise ValueError("User ID is required")
            
            user_context = await self.get_user_context(user_id)
            
            result = {
                "user_id": user_context["user_id"],
                "user_name": user_context["user_name"],
                "user_email": user_context["user_email"],
                "account_type": user_context["account_type"],
                "is_super_admin": user_context["is_super_admin"],
                "company": str(user_context["company_id"]) if user_context["company_id"] else None,
                "permissions": user_context["permissions"]
            }
            
            return f"User permissions: {json.dumps(result, default=str, indent=2)}"
            
        except Exception as e:
            return f"Error getting user permissions: {str(e)}"


class MongoDBListCollectionsInput(BaseModel):
    dummy: str = Field(description="Dummy field, not used")


class MongoDBListCollectionsTool(BaseMongoDBTool):
    name: str = "mongodb_list_collections"
    description: str = "List all collections in the database"
    args_schema: Type[BaseModel] = MongoDBListCollectionsInput

    def _run(self, dummy: str) -> str:
        return "This tool only supports async operations"

    async def _arun(self, dummy: str) -> str:
        try:
            collections = await db.list_collection_names()
            return f"Available collections: {', '.join(collections)}"
        except Exception as e:
            return f"Error listing collections: {str(e)}"


class MongoDBGetSchemaInput(BaseModel):
    collection: str = Field(description="The collection name to get schema for")


class MongoDBGetSchemaTool(BaseMongoDBTool):
    name: str = "mongodb_get_schema"
    description: str = "Get the schema of a collection by examining sample documents"
    args_schema: Type[BaseModel] = MongoDBGetSchemaInput

    def _run(self, collection: str) -> str:
        return "This tool only supports async operations"

    async def _arun(self, collection: str) -> str:
        try:
            coll = db[collection]
            sample = await coll.find_one()
            if sample:
                key_fields = {k: v for k, v in sample.items() if k in ['name', 'email', 'phone', 'status', 'amount', 'clientName']}
                return f"Key fields in {collection}: {json.dumps(key_fields, default=str)}"
            else:
                return f"Collection {collection} is empty"
        except Exception as e:
            return f"Error getting schema: {str(e)}"


class MongoDBGetSchemaKnowledgeInput(BaseModel):
    dummy: str = Field(description="Dummy field, not used")


class MongoDBGetSchemaKnowledgeTool(BaseMongoDBTool):
    name: str = "mongodb_get_schema_knowledge"
    description: str = "Get comprehensive knowledge about database schema, relationships, and query examples"
    args_schema: Type[BaseModel] = MongoDBGetSchemaKnowledgeInput

    def _run(self, dummy: str) -> str:
        return "This tool only supports async operations"

    async def _arun(self, dummy: str) -> str:
        return SCHEMA_KNOWLEDGE


# Create tools
tools = [
    EnhancedMongoDBQueryTool(),
    EnhancedMongoDBAggregationTool(),
    MongoDBGetUserPermissionsTool(),
    MongoDBListCollectionsTool(),
    MongoDBGetSchemaTool(),
    MongoDBGetSchemaKnowledgeTool()
]


# Create the ReAct prompt template
react_prompt = PromptTemplate.from_template("""You are a helpful AI assistant for MongoDB queries with user-based filtering. Tools: {tools} ONLY.

Use EXACT format ALWAYS. If insufficient info: "Thought: Insufficient information" and "Final Answer: Cannot provide answer due to missing data."

Question: {input}
Thought: Step-by-step reasoning. Verify schema/knowledge. No assumptions. Dynamically adapt queries/aggregations from schema fields and relationships.
Action: One of [{tool_names}]. mongodb_aggregation for complex; mongodb_query for simple lookups.
Action Input: Valid JSON using ONLY schema fields/relationships.
Observation: Result
... (Max 3 repeats. Use {agent_scratchpad}.)
Thought: Final answer ready based on observations.
Final Answer: Clean JSON with relevant fields ONLY.

GUIDELINES:
1. Build EVERY query/aggregation dynamically from schema knowledge (fields, relationships); no fixed templates.
2. mongodb_aggregation for grouping/counting/joining; adapt stages to exact schema.
3. mongodb_query ONLY for simple ID lookups.
4. Use $lookup with EXACT schema fields/foreign keys; no invented joins.
5. ALWAYS filter by CURRENT_USER_ID in FIRST $match; adapt to schema relationships.
6. Project ONLY meaningful fields if in schema.
7. Use exact ObjectId names; no guesses.
8. Restrict to user's data; reject if unfilterable.
9. No hallucinations; note missing fields in Thought.
10. Replace placeholders if provided; else stop.

SCHEMA KNOWLEDGE:
{schema_knowledge}

USER FILTERING:
- Dynamically filter by CURRENT_USER_ID/company via schema relationships ONLY.
- User queries: Match CURRENT_USER_ID first, adapt to schema.
- Company queries: Match USER_COMPANY_ID first, adapt to schema.
- $lookup with exact fields; project schema-only fields.
- Reject if impossible; exclude non-schema fields, note in Thought.

EXAMPLES (For guidance; adapt dynamically to schema):
- User's name: Simple query on users, match _id = CURRENT_USER_ID, project name.
- Assigned leads: Aggregate leads, match assignedTo = CURRENT_USER_ID, lookup users if needed, project name/status.

Scratchpad: {agent_scratchpad}
""").partial(schema_knowledge=SCHEMA_KNOWLEDGE)


# Create ReAct agent with the prompt
agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)


class OptimizedAgentExecutor:
    def __init__(self, agent_executor):
        self.agent_executor = agent_executor
        self.current_user_id = None
    
    async def ainvoke(self, inputs, config=None):
        # Set user context for this request
        self.current_user_id = inputs.get("user_id")
        
        if not self.current_user_id:
            return {
                "output": "Error: user_id is required for all queries",
                "status": "error"
            }
        
        # Validate user input for security
        user_input = inputs.get("input", "")
        if not validate_user_input(user_input):
            return {
                "output": "Error: Invalid input detected. Please avoid using potentially dangerous patterns.",
                "status": "error"
            }
        
        try:
            # Pre-validate user exists and cache context
            await user_context_manager.get_user_context(self.current_user_id)
            
            # Add user_id to all tool calls automatically
            enhanced_inputs = {**inputs, "user_id": self.current_user_id}
            
            result = await self.agent_executor.ainvoke(enhanced_inputs, config)
            return result
            
        except ValueError as e:
            return {
                "output": f"Authentication error: {str(e)}",
                "status": "error"
            }
        except Exception as e:
            return {
                "output": f"Execution error: {str(e)}",
                "status": "error"
            }

# Then update the initialization:
base_agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3,
    early_stopping_method="generate"
)

# Wrap it with our custom executor
agent_executor = OptimizedAgentExecutor(base_agent_executor)


async def process_chatbot_query(user_message: str, user_id: str) -> dict:
    """
    Process a chatbot query with proper error handling and validation
    """
    try:
        # Validate inputs
        if not user_message or not user_id:
            return {
                "status": "error",
                "message": "Both user_message and user_id are required"
            }
        
        if not ObjectId.is_valid(user_id):
            return {
                "status": "error", 
                "message": "Invalid user_id format"
            }
        
        # Execute query
        result = await agent_executor.ainvoke({
            "input": user_message,
            "user_id": user_id
        })
        
        return {
            "status": "success",
            "response": result.get("output", ""),
            "intermediate_steps": result.get("intermediate_steps", [])
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Query processing failed: {str(e)}"
        }


# Usage example and testing functions
async def test_chatbot():
    """Test function for the chatbot"""
    user_id = "67ee232ca893aaad92fe7214"  # Replace with actual user ID
    
    queries = [
        "Show me my assigned leads",
        "How many leads do I have?", 
        "Show my lead notes from this week",
        "What are my permissions?",
        "List all available collections"
    ]
    
    for query in queries:
        print(f"\n🤖 Processing: {query}")
        result = await process_chatbot_query(query, user_id)
        
        if result["status"] == "success":
            print(f"✅ Response: {result['response']}")
        else:
            print(f"❌ Error: {result['message']}")


# FastAPI integration example
def create_chatbot_endpoint():
    """Example FastAPI endpoint integration"""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    
    app = FastAPI(title="MongoDB Chatbot API")
    
    class ChatRequest(BaseModel):
        message: str
        user_id: str
    
    class ChatResponse(BaseModel):
        status: str
        response: str
        message: Optional[str] = None
    
    @app.post("/chat", response_model=ChatResponse)
    async def chat_endpoint(request: ChatRequest):
        result = await process_chatbot_query(request.message, request.user_id)
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return ChatResponse(
            status=result["status"],
            response=result["response"]
        )
    
    return app


# Main execution
if __name__ == "__main__":
    # Run test
    asyncio.run(test_chatbot())
