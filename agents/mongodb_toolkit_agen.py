import os
import json
import google.generativeai as genai
from config import settings
from database import db
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from typing import Optional, Type, Any, Dict
from pydantic import BaseModel, Field
from bson import ObjectId

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
    Recursively convert string values to ObjectIds for specified fields
    """
    result = {}
    for key, value in query_dict.items():
        if key in objectid_fields and isinstance(value, str):
            try:
                result[key] = ObjectId(value)
            except Exception:
                # If conversion fails, keep as string
                result[key] = value
        elif isinstance(value, dict):
            result[key] = convert_to_objectids(value, objectid_fields)
        elif isinstance(value, list):
            result[key] = [
                convert_to_objectids(item, objectid_fields) if isinstance(item, dict) else item
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

class MongoDBQueryInput(BaseModel):
    query: str = Field(description="The MongoDB query in JSON format")
    user_id: Optional[str] = Field(description="User ID for permission-based filtering", default=None)
    collection: Optional[str] = Field(description="Collection name", default=None)
    count: Optional[bool] = Field(description="Whether to count documents", default=False)
    projection: Optional[str] = Field(description="Projection fields", default=None)
    limit: Optional[int] = Field(description="Limit number of results", default=None)

class MongoDBQueryTool(BaseTool):
    name: str = "mongodb_query"
    description: str = "Execute simple MongoDB queries for basic lookups with read permission checks."
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
            global current_user_context

            # Get user_id from context if not provided
            if not user_id:
                user_id = current_user_context.get("user_id")

            # Validate user_id
            if not user_id or not isinstance(user_id, str) or not ObjectId.is_valid(user_id):
                return {
                    "status": "error",
                    "error_code": "InvalidUserID",
                    "message": f"Invalid or missing user_id: {user_id}"
                }
            user_obj_id = ObjectId(user_id)

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
                            try:
                                projection_dict = json.loads(proj_raw)
                            except json.JSONDecodeError:
                                return {
                                    "status": "error",
                                    "error_code": "InvalidProjection",
                                    "message": "Projection is not valid JSON."
                                }
                        elif isinstance(proj_raw, dict):
                            projection_dict = proj_raw
                    q_user_id = query_data.get("user_id")
                    if q_user_id and isinstance(q_user_id, str) and ObjectId.is_valid(q_user_id):
                        user_obj_id = ObjectId(q_user_id)
                        user_id = q_user_id
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

            # Fetch user document to get company ID
            user = await db.users.find_one({"_id": user_obj_id})
            if not user:
                return {
                    "status": "error",
                    "error_code": "UserNotFound",
                    "message": f"User with ID {user_id} not found."
                }

            # Get company ID from user document
            user_company = user.get("company")
            if user_company and not isinstance(user_company, ObjectId):
                user_company = ObjectId(user_company)

            # Replace placeholders in query using fetched company ID
            mongo_query = replace_placeholders_recursively(mongo_query, user_id, user_company)

            # Convert ObjectId fields AFTER placeholder replacement
            if collection_name in OBJECTID_FIELDS:
                mongo_query = convert_to_objectids(mongo_query, OBJECTID_FIELDS[collection_name])

            # Permission and filtering logic
            account_type = user.get("account_type", "")
            user_permissions = user.get("permissions", {})
            is_super_admin = account_type == "company_super_admin"

            # Permission check
            perms = user_permissions.get(collection_name, {})
            has_read_permission = perms and ("read" in perms)
            
            if not is_super_admin and not has_read_permission:
                return {
                    "status": "error",
                    "error_code": "PermissionDenied",
                    "message": f"You do not have read permission on the collection '{collection_name}'."
                }

            # Apply company filtering for sub-admins
            if account_type == "company_sub_admin" and user_company:
                if "company" not in mongo_query:
                    mongo_query["company"] = user_company
                else:
                    requested_company = mongo_query.get("company")
                    if isinstance(requested_company, str) and ObjectId.is_valid(requested_company):
                        requested_company = ObjectId(requested_company)
                        mongo_query["company"] = requested_company
                    if requested_company != user_company:
                        return {
                            "status": "error",
                            "error_code": "PermissionDenied",
                            "message": "Access denied to company data outside your assigned company.",
                        }

            # Special handling for users collection
            if collection_name == "users":
                mongo_query = {"_id": user_obj_id}

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
                    projection_dict = {"name": 1, "email": 1, "phone": 1, "leadStatus": 1, "_id": 0}
                elif collection_name == "users":
                    projection_dict = {"name": 1, "email": 1, "_id": 0}
                elif collection_name == "companies":
                    projection_dict = {"name": 1, "clientName": 1, "_id": 0}
                elif collection_name == "properties":
                    projection_dict = {"name": 1, "_id": 0}
                elif collection_name == "rent-payments":
                    projection_dict = {"amount": 1, "_id": 0}
                else:
                    projection_dict = {"name": 1, "_id": 0}

            cursor = coll.find(mongo_query, projection_dict)
            if limit_val:
                cursor = cursor.limit(limit_val)

            documents = await cursor.to_list(length=limit_val or 50)

            if not documents:
                return {
                    "status": "success",
                    "count": 0,
                    "data": [],
                    "message": f"No {collection_name} found matching the criteria."
                }

            return {
                "status": "success",
                "count": len(documents),
                "data": documents,
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
                "message": "An error occurred while executing the query.",
            }

class MongoDBAggregationInput(BaseModel):
    pipeline: str = Field(description="The MongoDB aggregation pipeline in JSON format")
    user_id: Optional[str] = Field(description="User ID for permission-based filtering", default=None)
    collection: Optional[str] = Field(description="Collection name", default=None)
    stages: Optional[str] = Field(description="Aggregation stages", default=None)

class MongoDBAggregationTool(BaseTool):
    name: str = "mongodb_aggregation"
    description: str = "Execute MongoDB aggregation pipelines for complex queries. ALWAYS include relevant fields from related collections using $lookup and $project. Use for queries like 'total tenants', 'leads by status', 'rent payments with tenant details', etc. Automatically filters by user's company."
    args_schema: Type[BaseModel] = MongoDBAggregationInput

    def _run(self, pipeline: str, user_id: Optional[str] = None) -> str:
        return "This tool only supports async operations"

    async def _arun(self, pipeline: str, user_id: Optional[str] = None, collection: Optional[str] = None, stages: Optional[str] = None) -> str:
        try:
            global current_user_context
            if not user_id:
                user_id = current_user_context.get("user_id")
            
            # Parse the aggregation pipeline
            if isinstance(pipeline, str):
                try:
                    pipeline_data = json.loads(pipeline)
                    collection_name = pipeline_data.get("collection") or collection
                    stages = pipeline_data.get("pipeline", []) or pipeline_data.get("stages", [])
                    if isinstance(stages, str):
                        try:
                            stages = json.loads(stages)
                        except json.JSONDecodeError:
                            return "Error: Invalid stages JSON"
                    if not isinstance(stages, list):
                        stages = []
                    user_id = pipeline_data.get("user_id") or user_id
                except json.JSONDecodeError:
                    collection_name = pipeline
                    stages = []
            else:
                collection_name = collection
                stages = stages or []
                if isinstance(stages, str):
                    try:
                        stages = json.loads(stages)
                    except json.JSONDecodeError:
                        return "Error: Invalid stages JSON"
                if not isinstance(stages, list):
                    stages = []
            
            if not collection_name:
                return "Error: collection name is required"
            
            # Fetch user document to get company ID
            if user_id:
                user = await db.users.find_one({"_id": ObjectId(user_id)})
                if not user:
                    return f"Error: User with ID {user_id} not found"
                
                # Get company ID from user document
                user_company = user.get("company")
                if user_company and not isinstance(user_company, ObjectId):
                    user_company = ObjectId(user_company)
                
                account_type = user.get("account_type", "")
                user_permissions = user.get("permissions", {})
                is_super_admin = account_type == "company_super_admin"
                
                # Replace placeholders in the entire pipeline using fetched company ID
                stages = replace_placeholders_recursively(stages, user_id, user_company)
                
                # Apply company filtering for non-super admins
                if not is_super_admin and user_company:
                    has_user_filter = any(
                        isinstance(stage, dict) and "$match" in stage and 
                        any(field in stage.get("$match", {}) for field in ["_id", "assignedTo", "assignee", "company"])
                        for stage in stages
                    )
                    
                    if not has_user_filter:
                        if collection_name == "users":
                            user_filter = {"$match": {"_id": ObjectId(user_id)}}
                        elif collection_name in ["leads", "lead-assignments", "lead-rotations"]:
                            user_filter = {"$match": {"assignedTo": ObjectId(user_id)}}
                        else:
                            user_filter = {"$match": {"company": user_company}}
                        stages.insert(0, user_filter)
                
                # Permission check
                collection_perms = user_permissions.get(collection_name, {})
                has_permission = (
                    "read" in collection_perms or 
                    "write" in collection_perms or 
                    is_super_admin
                )
                
                if not has_permission:
                    return f"Error: You don't have permission to access {collection_name}."
            
            # Convert ObjectIds AFTER placeholder replacement
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

class MongoDBGetUserPermissionsInput(BaseModel):
    user_id: str = Field(description="The user ID to get permissions for")

class MongoDBGetUserPermissionsTool(BaseTool):
    name: str = "mongodb_get_user_permissions"
    description: str = "Get user permissions for filtering queries"
    args_schema: Type[BaseModel] = MongoDBGetUserPermissionsInput

    def _run(self, user_id: str) -> str:
        return "This tool only supports async operations"

    async def _arun(self, user_id: Optional[str] = None) -> str:
        try:
            global current_user_context
            
            # Fix: Properly get user_id from context if not provided
            if not user_id:
                user_id = current_user_context.get("user_id")
                if not user_id:
                    return "Error: No user_id provided"
            
            # Fix: Validate user_id before creating ObjectId
            if not isinstance(user_id, str) or not ObjectId.is_valid(user_id):
                return f"Error: Invalid user_id format: {user_id}"
            
            # Get user details
            user = await db.users.find_one({"_id": ObjectId(user_id)})
            if not user:
                return f"Error: User with ID {user_id} not found"
            
            permissions = user.get("permissions", {})
            account_type = user.get("account_type", "")
            is_super_admin = account_type == "company_super_admin"
            company = user.get("company")
            
            result = {
                "user_id": user_id,
                "user_name": user.get("name"),
                "user_email": user.get("email"),
                "account_type": account_type,
                "is_super_admin": is_super_admin,
                "company": str(company) if company else None,
                "permissions": permissions
            }
            
            return f"User permissions: {json.dumps(result, default=str, indent=2)}"
            
        except Exception as e:
            return f"Error getting user permissions: {str(e)}"

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
                key_fields = {k: v for k, v in sample.items() if k in ['name', 'email', 'phone', 'status', 'amount', 'clientName']}
                return f"Key fields in {collection}: {json.dumps(key_fields, default=str)}"
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
""")

# Create ReAct agent with the prompt
agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)

# Global context to store current user_id
current_user_context = {"user_id": None}

# Create custom agent executor that automatically passes user_id to tools
class CustomAgentExecutor(AgentExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    async def ainvoke(self, inputs, config=None):
        # Extract user_id from inputs and store it globally
        global current_user_context
        user_id = inputs.get("user_id")
        current_user_context["user_id"] = user_id
        if user_id:
            print(f"Debug: Set current_user_context to {user_id}")  # Debug line
        return await super().ainvoke(inputs, config)

agent_executor = CustomAgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)
