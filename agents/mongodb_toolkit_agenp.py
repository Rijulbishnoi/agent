import os
import json
import re
import time
import asyncio
from datetime import datetime, timedelta
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
from contextvars import ContextVar

current_user_context = ContextVar('current_user_context', default={"user_id": None})

def set_current_user_id(user_id: str):
    """Set user ID in context"""
    current_user_context.set({"user_id": user_id})

def get_current_user_id() -> Optional[str]:
    """Get user ID from context"""
    context = current_user_context.get()
    return context.get("user_id") if context else None
 
# Configure Google Gemini API key
genai.configure(api_key=settings.gemini_api_key)

# Initialize Gemini LLM wrapper
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
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


Company Filtering:
- All data is filtered by the user's company automatically
- Users can only access data belonging to their company
- Company ID is automatically injected into all queries


Field Value Clarifications:
- 'leadStatus' tracks stages such as 'New', 'In Progress', 'Closed - Won', 'Closed - Lost'.
- 'status' field indicates general activity, e.g., 'active', 'inactive'.


Example Queries:
- "Show company leads": Query leads where company = USER_COMPANY_ID
- "Show notes for a lead": Query lead-notes where lead = lead_id and company = USER_COMPANY_ID
- "Show properties": Query properties where company = USER_COMPANY_ID
- "Show company users": Query users where company = USER_COMPANY_ID
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

# Global context manager
user_context_manager = UserContextManager()

# Pydantic Models
class MongoDBQueryInput(BaseModel):
    query: str = Field(description="The MongoDB query in JSON format")
    user_id: Optional[str] = Field(description="User ID for company-based filtering", default=None)
    collection: Optional[str] = Field(description="Collection name", default=None)
    count: Optional[bool] = Field(description="Whether to count documents", default=False)
    projection: Optional[str] = Field(description="Projection fields", default=None)
    limit: Optional[int] = Field(description="Limit number of results", default=None)

class MongoDBQueryTool(BaseTool):
    name: str = "mongodb_query"
    description: str = "Execute MongoDB queries with company-based filtering"
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
        try:
            # CRITICAL FIX: Get user_id from context if not provided
            effective_user_id = user_id or get_current_user_id()
            
            if not effective_user_id:
                return {
                    "status": "error",
                    "error_code": "MissingUserID",
                    "message": "User ID is required and not found in context"
                }

            # Validate effective_user_id
            if not ObjectId.is_valid(effective_user_id):
                return {
                    "status": "error",
                    "error_code": "InvalidUserID",
                    "message": f"Invalid user_id format: {effective_user_id}"
                }

            # Parse query parameters
            collection_name = collection
            mongo_query = {}
            count_flag = count
            limit_val = limit
            projection_dict = None

            # Parse JSON query if provided
            if isinstance(query, str):
                try:
                    query_data = json.loads(query)
                    collection_name = query_data.get("collection") or collection_name
                    mongo_query = query_data.get("query", {})
                    limit_val = query_data.get("limit") or limit_val
                    count_flag = query_data.get("count", False) or count_flag
                    proj_raw = query_data.get("projection") or projection
                    if proj_raw:
                        if isinstance(proj_raw, str):
                            projection_dict = json.loads(proj_raw)
                        elif isinstance(proj_raw, dict):
                            projection_dict = proj_raw
                except json.JSONDecodeError:
                    if not collection_name:
                        collection_name = query

            if not collection_name:
                return {
                    "status": "error",
                    "error_code": "MissingCollection",
                    "message": "Collection name is required."
                }

            # Check if collection exists
            existing_collections = await db.list_collection_names()
            if collection_name not in existing_collections:
                return {
                    "status": "error",
                    "error_code": "InvalidCollection",
                    "message": f"Collection '{collection_name}' does not exist."
                }

            # Get user context for company filtering
            user_context = await user_context_manager.get_user_context(effective_user_id)
            user_company = user_context.get("company_id")

            if not user_company:
                return {
                    "status": "error",
                    "error_code": "NoCompany",
                    "message": "User is not associated with any company."
                }

            # Replace placeholders in query
            mongo_query = replace_placeholders_recursively(mongo_query, effective_user_id, user_company)

            # Convert ObjectId fields
            if collection_name in OBJECTID_FIELDS:
                mongo_query = convert_to_objectids(mongo_query, OBJECTID_FIELDS[collection_name])

            # Apply company filtering automatically
            if collection_name != "companies":
                if "company" not in mongo_query:
                    mongo_query["company"] = ObjectId(user_company)

            coll = db[collection_name]

            if count_flag:
                total_count = await coll.count_documents(mongo_query)
                return {
                    "status": "success",
                    "count": total_count
                }

            # Apply default projections if none specified
            if not projection_dict:
                if collection_name == "leads":
                    projection_dict = {"name": 1, "email": 1, "phone": 1, "leadStatus": 1, "_id": 1}
                elif collection_name == "users":
                    projection_dict = {"name": 1, "email": 1, "designation": 1, "_id": 1}
                elif collection_name == "companies":
                    projection_dict = {"name": 1, "clientName": 1, "_id": 1}
                elif collection_name == "properties":
                    projection_dict = {"name": 1, "_id": 1}
                else:
                    projection_dict = {"name": 1, "_id": 1}

            cursor = coll.find(mongo_query, projection_dict)
            if limit_val:
                cursor = cursor.limit(limit_val)

            documents = await cursor.to_list(length=limit_val or 50)

            return {
                "status": "success",
                "count": len(documents),
                "data": documents,
                "message": f"Found {len(documents)} {collection_name}" if documents else "No documents found"
            }

        except json.JSONDecodeError:
            return {
                "status": "error",
                "error_code": "InvalidJSON",
                "message": "Invalid JSON format in input query or projection.",
            }
        except Exception as e:
            return {
                "status": "error",
                "error_code": "ExecutionError",
                "message": f"Query execution failed: {str(e)}",
            }

class MongoDBAggregationInput(BaseModel):
    pipeline: str = Field(description="The MongoDB aggregation pipeline in JSON format")
    user_id: Optional[str] = Field(description="User ID for company-based filtering", default=None)
    collection: Optional[str] = Field(description="Collection name", default=None)
    stages: Optional[str] = Field(description="Aggregation stages", default=None)

class MongoDBAggregationTool(BaseTool):
    name: str = "mongodb_aggregation"
    description: str = "Execute MongoDB aggregation pipelines with automatic company filtering"
    args_schema: Type[BaseModel] = MongoDBAggregationInput

    def _run(self, pipeline: str, user_id: Optional[str] = None) -> str:
        return "This tool only supports async operations"

    async def _arun(self, pipeline: str, user_id: Optional[str] = None, 
                   collection: Optional[str] = None, stages: Optional[str] = None) -> str:
        try:
            # CRITICAL FIX: Get user_id from context if not provided
            effective_user_id = user_id or get_current_user_id()
            
            if not effective_user_id:
                return "Error: User ID not found in context"
            
            # Parse the aggregation pipeline
            if isinstance(pipeline, str):
                try:
                    pipeline_data = json.loads(pipeline)
                    collection_name = pipeline_data.get("collection") or collection
                    stages = pipeline_data.get("pipeline", []) or pipeline_data.get("stages", [])
                    if isinstance(stages, str):
                        stages = json.loads(stages)
                    if not isinstance(stages, list):
                        stages = []
                except json.JSONDecodeError:
                    collection_name = pipeline
                    stages = []
            else:
                collection_name = collection
                stages = stages or []
                if isinstance(stages, str):
                    stages = json.loads(stages)

            if not collection_name:
                return "Error: collection name is required"

            # Get user context for company filtering
            user_context = await user_context_manager.get_user_context(effective_user_id)
            user_company = user_context.get("company_id")

            if not user_company:
                return "Error: User is not associated with any company"

            # Replace placeholders in the entire pipeline
            stages = replace_placeholders_recursively(stages, effective_user_id, user_company)

            # Apply company filtering - add $match stage at the beginning if not exists
            has_company_filter = any(
                isinstance(stage, dict) and "$match" in stage and 
                "company" in stage.get("$match", {})
                for stage in stages
            )

            if not has_company_filter and collection_name != "companies":
                if collection_name == "users":
                    company_filter = {"$match": {"company": ObjectId(user_company)}}
                else:
                    company_filter = {"$match": {"company": ObjectId(user_company)}}
                stages.insert(0, company_filter)

            # Convert ObjectIds
            if collection_name in OBJECTID_FIELDS:
                for i, stage in enumerate(stages):
                    if isinstance(stage, dict):
                        stages[i] = convert_to_objectids(stage, OBJECTID_FIELDS[collection_name])

            # Execute aggregation
            collection = db[collection_name]
            cursor = collection.aggregate(stages)
            results = await cursor.to_list(length=100)

            if not results:
                return f"No results found for aggregation on {collection_name}."

            if len(results) == 1:
                return f"Aggregation result: {json.dumps(results[0], default=str, indent=2)}"
            else:
                return f"Aggregation results ({len(results)} items): {json.dumps(results, default=str, indent=2)}"

        except json.JSONDecodeError:
            return "Error: Invalid JSON format"
        except Exception as e:
            return f"Error executing aggregation: {str(e)}"

class MongoDBGetUserInfoInput(BaseModel):
    user_id: str = Field(description="The user ID to get info for")

class MongoDBGetUserInfoTool(BaseTool):
    name: str = "mongodb_get_user_info"
    description: str = "Get user information and company details"
    args_schema: Type[BaseModel] = MongoDBGetUserInfoInput

    def _run(self, user_id: str) -> str:
        return "This tool only supports async operations"

    async def _arun(self, user_id: Optional[str] = None) -> str:
        try:
            effective_user_id = user_id or get_current_user_id()
            if not effective_user_id:
                return "Error: User ID not found in context"
            
            if not ObjectId.is_valid(effective_user_id):
                return f"Error: Invalid user_id format: {effective_user_id}"

            user_context = await user_context_manager.get_user_context(effective_user_id)

            result = {
                "user_id": user_context["user_id"],
                "user_name": user_context["user_name"],
                "user_email": user_context["user_email"],
                "company_id": str(user_context["company_id"]) if user_context["company_id"] else None
            }

            return f"User info: {json.dumps(result, default=str, indent=2)}"

        except Exception as e:
            return f"Error getting user info: {str(e)}"

class MongoDBListCollectionsInput(BaseModel):
    dummy: str = Field(description="Dummy field, not used")

class MongoDBListCollectionsTool(BaseTool):
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

class MongoDBGetSchemaTool(BaseTool):
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
                return f"Sample document from {collection}: {json.dumps(sample, default=str, indent=2)}"
            else:
                return f"Collection {collection} is empty"
        except Exception as e:
            return f"Error getting schema: {str(e)}"

class MongoDBGetSchemaKnowledgeInput(BaseModel):
    dummy: str = Field(description="Dummy field, not used")

class MongoDBGetSchemaKnowledgeTool(BaseTool):
    name: str = "mongodb_get_schema_knowledge"
    description: str = "Get comprehensive knowledge about database schema, relationships, and query examples"
    args_schema: Type[BaseModel] = MongoDBGetSchemaKnowledgeInput

    def _run(self, dummy: str) -> str:
        return "This tool only supports async operations"

    async def _arun(self, dummy: str) -> str:
        return SCHEMA_KNOWLEDGE

# Create tools
tools = [
    MongoDBQueryTool(),
    MongoDBAggregationTool(),
    MongoDBGetUserInfoTool(),
    MongoDBListCollectionsTool(),
    MongoDBGetSchemaTool(),
    MongoDBGetSchemaKnowledgeTool()
]

# Create the ReAct prompt template
react_prompt = PromptTemplate.from_template("""You are a helpful AI assistant for MongoDB queries with company-based filtering. Tools: {tools} ONLY.

Use EXACT format ALWAYS. If insufficient info: "Thought: Insufficient information" and "Final Answer: Cannot provide answer due to missing data."

Question: {input}
Thought: Step-by-step reasoning. Verify schema/knowledge. No assumptions. Dynamically adapt queries/aggregations from schema fields and relationships.
Action: One of [{tool_names}]. mongodb_aggregation for complex; mongodb_query for simple lookups.
Action Input: Valid JSON using ONLY schema fields/relationships. Do NOT include user_id - it's handled automatically.
Observation: Result
... (Max 3 repeats. Use {agent_scratchpad}.)
Thought: Final answer ready based on observations.
Final Answer: Clean JSON with relevant fields ONLY.

GUIDELINES:
1. Build EVERY query/aggregation dynamically from schema knowledge (fields, relationships); no fixed templates.
2. mongodb_aggregation for grouping/counting/joining; adapt stages to exact schema.
3. mongodb_query ONLY for simple ID lookups.
4. Use $lookup with EXACT schema fields/foreign keys; no invented joins.
5. ALWAYS filter by USER_COMPANY_ID automatically; all data is company-scoped.
6. Project ONLY meaningful fields if in schema.
7. Use exact ObjectId names; no guesses.
8. All queries are automatically filtered by company - no need to add company filter manually.
9. No hallucinations; note missing fields in Thought.
10. Replace placeholders if provided; else stop.

SCHEMA KNOWLEDGE:
{schema_knowledge}

COMPANY FILTERING:
- All data is automatically filtered by user's company
- USER_COMPANY_ID is automatically injected into queries
- Users can only see data from their own company
- No manual company filtering needed in queries

EXAMPLES (For guidance; adapt dynamically to schema):
- "Show leads": Action Input: {{"collection": "leads", "query": "{{}}"}}
- "Count users": Action Input: {{"collection": "users", "pipeline": [{{"$count": "totalUsers"}}]}}
- "Show lead assignments": Action Input: {{"collection": "lead-assignments", "query": "{{}}"}}

Scratchpad: {agent_scratchpad}
""").partial(schema_knowledge=SCHEMA_KNOWLEDGE)

# Create ReAct agent with the prompt
agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)

class OptimizedAgentExecutor:
    def __init__(self, agent_executor):
        self.agent_executor = agent_executor
        self.current_user_id = None
    
    async def ainvoke(self, inputs, config=None):
        # CRITICAL: Set user context BEFORE any tool execution
        self.current_user_id = inputs.get("user_id")
        
        if not self.current_user_id:
            return {
                "output": "Error: user_id is required for all queries",
                "status": "error"
            }
        
        # CRITICAL FIX: Set global context immediately
        set_current_user_id(self.current_user_id)
        
        try:
            # Pre-validate user exists
            await user_context_manager.get_user_context(self.current_user_id)
            
            # Execute agent with enhanced inputs
            enhanced_inputs = {**inputs, "user_id": self.current_user_id}
            result = await self.agent_executor.ainvoke(enhanced_inputs, config)
            return result
            
        except Exception as e:
            return {
                "output": f"Execution error: {str(e)}",
                "status": "error"
            }

# Initialize agent executor
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

# Test function
async def test_chatbot():
    """Test function for the chatbot"""
    user_id = "67ee232ca893aaad92fe7214"  # Replace with actual user ID
    
    queries = [
        "How many leads does our company have?",
        "Show me all company users", 
        "Count properties in our company",
        "Show company lead assignments",
        "What collections are available?"
    ]
    
    for query in queries:
        print(f"\n Processing: {query}")
        result = await process_chatbot_query(query, user_id)
        
        if result["status"] == "success":
            print(f" Response: {result['response']}")
        else:
            print(f" Error: {result['message']}")


# Main execution
if __name__ == "__main__":
    # Run test
    asyncio.run(test_chatbot())
