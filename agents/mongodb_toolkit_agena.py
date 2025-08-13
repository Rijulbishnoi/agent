# mongodb_toolkit_agen.py - Final Working Version (No Template Errors)
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
import logging

# Setup logging
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
    Recursively convert string values to ObjectIds for specified fields
    """
    result = {}
    for key, value in query_dict.items():
        if key in objectid_fields and isinstance(value, str):
            try:
                result[key] = ObjectId(value)
            except Exception:
                result[key] = value
        elif key in objectid_fields and isinstance(value, dict):
            if "$oid" in value and isinstance(value["$oid"], str):
                try:
                    result[key] = ObjectId(value["$oid"])
                except Exception:
                    result[key] = value
            else:
                result[key] = convert_to_objectids(value, objectid_fields)
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
    "users": ["_id", "company", "designation", "groups"],
    "companies": ["_id", "country", "plan", "superAdmin"],
    "properties": ["_id", "bhk", "bhkType", "company", "project", "propertyUnitSubType"],
    "projects": ["_id", "company", "land", "category", "country"],
    "brokers": ["_id", "company", "country"],
    "tenants": ["_id", "company", "project", "property"],
    "rent-payments": ["_id", "company", "project", "property", "tenant"],
    "property-bookings": ["_id", "company", "lead", "project", "property", "broker"],
    "property-payments": ["_id", "company", "project", "property", "booking"],
    "broker-payments": ["_id", "company", "broker", "lead", "project", "property"],
    "teams": ["_id", "company"],
    "designations": ["_id", "company"],
    "bhk": ["_id", "company"],
    "bhk-types": ["_id", "company"],
    "project-categories": ["_id", "company"],
    "property-unit-sub-types": ["_id", "company"],
    "amenities": ["_id", "company"],
    "lands": ["_id", "company", "country"],
    "tags": ["_id", "company"],
    "settings": ["_id", "company"],
    "attendance": ["_id", "company", "user"],
    "contracts": ["_id", "company", "project"]
}

# Helper function to get user company
async def get_user_company(user_id: str):
    """Get user's company ID - CRITICAL for company filtering"""
    if not user_id or not ObjectId.is_valid(user_id):
        logger.warning(f"Invalid user_id: {user_id}")
        return None
    
    try:
        user = await db.users.find_one({"_id": ObjectId(user_id)})
        if not user:
            logger.warning(f"User not found: {user_id}")
            return None
        
        company = user.get("company")
        if company and not isinstance(company, ObjectId):
            company = ObjectId(company)
        
        logger.info(f"User {user_id} belongs to company: {company}")
        return company
    except Exception as e:
        logger.error(f"Error getting user company: {e}")
        return None

# Enhanced MongoDBQueryTool with STRICT company filtering
class MongoDBQueryInput(BaseModel):
    query: str = Field(description="The MongoDB query in JSON format", default="")
    user_id: Optional[str] = Field(description="User ID for company filtering", default=None)
    collection: Optional[str] = Field(description="Collection name", default=None)
    count: Optional[bool] = Field(description="Whether to count documents", default=False)
    projection: Optional[str] = Field(description="Projection fields", default=None)
    limit: Optional[int] = Field(description="Limit number of results", default=50)

class MongoDBQueryTool(BaseTool):
    name: str = "mongodb_query"
    description: str = "Execute simple MongoDB queries with MANDATORY company filtering and intelligent field detection."
    args_schema: Type[BaseModel] = MongoDBQueryInput

    def _run(self, *args, **kwargs) -> str:
        return "This tool only supports async operations"

    async def _arun(
        self,
        query: str = "",
        user_id: Optional[str] = None,
        collection: Optional[str] = None,
        count: Optional[bool] = False,
        projection: Optional[str] = None,
        limit: Optional[int] = 50,
    ) -> dict:
        try:
            global current_user_context
            
            # CRITICAL: Get user_id from context if not provided
            if not user_id:
                user_id = current_user_context.get("user_id")
            
            logger.info(f"Processing query with user_id: {user_id}")
            
            # MANDATORY: Get user's company - NO QUERIES WITHOUT COMPANY FILTER
            if not user_id:
                return {
                    "status": "error",
                    "error_code": "MissingUserContext",
                    "message": "User ID is required for company-specific data access."
                }
            
            user_company = await get_user_company(user_id)
            if not user_company:
                return {
                    "status": "error",
                    "error_code": "InvalidUserOrCompany",
                    "message": "Could not determine user's company. Access denied."
                }

            # Parse query parameters
            collection_name = collection
            mongo_query = {}
            count_flag = count
            limit_val = limit
            projection_dict = None

            # Parse JSON query if provided
            if query and query.strip():
                try:
                    query_data = json.loads(query)
                    collection_name = query_data.get("collection") or collection_name
                    mongo_query = query_data.get("query", query_data.get("filter", {}))
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
                        collection_name = query.strip()

            if not collection_name:
                return {
                    "status": "error",
                    "error_code": "MissingCollection",
                    "message": "Collection name is required."
                }

            # Check if collection exists
            try:
                existing_collections = await db.list_collection_names()
                if collection_name not in existing_collections:
                    return {
                        "status": "error",
                        "error_code": "InvalidCollection",
                        "message": f"Collection '{collection_name}' does not exist."
                    }
            except Exception as e:
                logger.warning(f"Could not list collections: {e}")

            # Replace placeholders in query
            mongo_query = replace_placeholders_recursively(mongo_query, user_id, user_company)

            # **CRITICAL COMPANY FILTERING - APPLIED TO ALL COLLECTIONS**
            collections_with_company = [
                "leads", "lead-assignments", "lead-notes", "lead-rotations", 
                "lead-visited-properties", "properties", "projects", "brokers", 
                "tenants", "rent-payments", "property-bookings", "property-payments",
                "broker-payments", "teams", "designations", "bhk", "bhk-types",
                "project-categories", "property-unit-sub-types", "amenities",
                "lands", "tags", "settings", "attendance", "contracts"
            ]
            
            # FORCE COMPANY FILTER ON ALL RELEVANT COLLECTIONS
            if collection_name in collections_with_company:
                # ALWAYS override any existing company filter with user's company
                mongo_query["company"] = user_company
                logger.info(f"FORCED company filter: {user_company}")
            
            # Special handling for users collection - only same company users
            elif collection_name == "users":
                mongo_query["company"] = user_company
                logger.info(f"FORCED company filter for users: {user_company}")
            
            # For companies collection - only user's own company
            elif collection_name == "companies":
                mongo_query["_id"] = user_company
                logger.info(f"FORCED company ID filter: {user_company}")

            # Convert ObjectId fields
            if collection_name in OBJECTID_FIELDS:
                mongo_query = convert_to_objectids(mongo_query, OBJECTID_FIELDS[collection_name])

            logger.info(f"Final query with MANDATORY company filter: {mongo_query}")

            coll = db[collection_name]

            if count_flag:
                total_count = await coll.count_documents(mongo_query)
                return {
                    "status": "success",
                    "count": total_count,
                    "company_id": str(user_company),
                    "company_filtered": True
                }

            # Apply intelligent projections
            if not projection_dict:
                if collection_name == "leads":
                    projection_dict = {"name": 1, "email": 1, "phone": 1, "leadStatus": 1, "sourceType": 1, "maxBudget": 1, "company": 1, "createdAt": 1, "_id": 1}
                elif collection_name == "users":
                    projection_dict = {"fullName": 1, "email": 1, "phone": 1, "designation": 1, "accountType": 1, "company": 1, "_id": 1}
                elif collection_name == "properties":
                    projection_dict = {"flatNo": 1, "blockName": 1, "propertyType": 1, "propertyStatus": 1, "minBudget": 1, "maxBudget": 1, "company": 1, "_id": 1}
                elif collection_name == "property-bookings":
                    projection_dict = {"lead": 1, "property": 1, "bookingAmount": 1, "bookingDate": 1, "bookingPaymentStatus": 1, "company": 1, "_id": 1}
                elif collection_name == "brokers":
                    projection_dict = {"name": 1, "email": 1, "phone": 1, "commissionPercent": 1, "status": 1, "company": 1, "_id": 1}
                else:
                    projection_dict = {"name": 1, "company": 1, "status": 1, "createdAt": 1, "_id": 1}

            cursor = coll.find(mongo_query, projection_dict)
            
            # Apply sorting (recent first for time-based data)
            if collection_name not in ["countries", "plans", "permissions"]:
                cursor = cursor.sort("createdAt", -1)
            
            if limit_val:
                cursor = cursor.limit(limit_val)

            documents = await cursor.to_list(length=limit_val or 50)

            return {
                "status": "success",
                "count": len(documents),
                "data": documents,
                "company_id": str(user_company),
                "company_filtered": True,
                "message": f"Retrieved {len(documents)} records for your company from {collection_name} collection."
            }

        except Exception as e:
            logger.error(f"ERROR in mongodb_query: {str(e)}")
            return {
                "status": "error",
                "error_code": "ExecutionError",
                "message": f"Query execution failed: {str(e)}",
            }

# Enhanced MongoDBAggregationTool with STRICT company filtering
class MongoDBAggregationInput(BaseModel):
    pipeline: str = Field(description="The MongoDB aggregation pipeline in JSON format")
    user_id: Optional[str] = Field(description="User ID for company filtering", default=None)
    collection: Optional[str] = Field(description="Collection name", default=None)
    stages: Optional[str] = Field(description="Aggregation stages", default=None)

class MongoDBAggregationTool(BaseTool):
    name: str = "mongodb_aggregation"
    description: str = "Execute MongoDB aggregation with MANDATORY company filtering and intelligent relationship handling."
    args_schema: Type[BaseModel] = MongoDBAggregationInput

    def _run(self, pipeline: str, user_id: Optional[str] = None) -> str:
        return "This tool only supports async operations"

    async def _arun(self, pipeline: str, user_id: Optional[str] = None, collection: Optional[str] = None, stages: Optional[str] = None) -> str:
        try:
            global current_user_context
            if not user_id:
                user_id = current_user_context.get("user_id")
            
            logger.info(f"Aggregation for user_id: {user_id}")
            
            # MANDATORY: Get user's company
            if not user_id:
                return "Error: User ID is required for company-specific data access."
            
            user_company = await get_user_company(user_id)
            if not user_company:
                return "Error: Could not determine user's company. Access denied."
            
            # Parse pipeline
            if pipeline and pipeline.strip():
                try:
                    pipeline_data = json.loads(pipeline)
                    collection_name = pipeline_data.get("collection") or collection
                    stages = pipeline_data.get("pipeline", []) or pipeline_data.get("stages", [])
                    if isinstance(stages, str):
                        stages = json.loads(stages)
                    if not isinstance(stages, list):
                        stages = []
                except json.JSONDecodeError:
                    collection_name = collection
                    stages = []
            else:
                collection_name = collection
                stages = stages or []
                if isinstance(stages, str):
                    stages = json.loads(stages)
                if not isinstance(stages, list):
                    stages = []
            
            if not collection_name:
                return "Error: Collection name is required"
            
            # Replace placeholders
            stages = replace_placeholders_recursively(stages, user_id, user_company)
            
            # **CRITICAL: FORCE COMPANY FILTERING ON ALL AGGREGATION STAGES**
            collections_with_company = [
                "leads", "lead-assignments", "lead-notes", "lead-rotations", 
                "lead-visited-properties", "properties", "projects", "brokers", 
                "tenants", "rent-payments", "property-bookings", "property-payments",
                "broker-payments", "teams", "designations", "bhk", "bhk-types",
                "project-categories", "property-unit-sub-types", "amenities",
                "lands", "tags", "settings", "attendance", "contracts"
            ]
            
            # FORCE company filter at the beginning of pipeline
            if collection_name in collections_with_company:
                # Remove any existing $match with company and add our own
                stages = [stage for stage in stages if not (
                    isinstance(stage, dict) and "$match" in stage and "company" in stage.get("$match", {})
                )]
                
                # Insert MANDATORY company filter at the beginning
                company_match = {"$match": {"company": user_company}}
                stages.insert(0, company_match)
                logger.info(f"FORCED company filter in aggregation: {user_company}")

            elif collection_name == "users":
                # For users, filter by company
                company_match = {"$match": {"company": user_company}}
                stages.insert(0, company_match)

            # **CRITICAL: Filter ALL $lookup stages by company**
            for i, stage in enumerate(stages):
                if isinstance(stage, dict) and "$lookup" in stage:
                    lookup_stage = stage["$lookup"]
                    lookup_from = lookup_stage.get("from")
                    
                    if lookup_from in collections_with_company:
                        # Force company filtering in lookups
                        if "pipeline" not in lookup_stage:
                            # Convert to pipeline-based lookup
                            local_field = lookup_stage.get("localField")
                            foreign_field = lookup_stage.get("foreignField")
                            as_field = lookup_stage.get("as")
                            
                            if local_field and foreign_field and as_field:
                                new_lookup = {
                                    "$lookup": {
                                        "from": lookup_from,
                                        "let": {f"local_{local_field}": f"${local_field}"},
                                        "pipeline": [
                                            {"$match": {"company": user_company}},
                                            {"$match": {"$expr": {"$eq": [f"${foreign_field}", f"$$local_{local_field}"]}}}
                                        ],
                                        "as": as_field
                                    }
                                }
                                stages[i] = new_lookup
                        else:
                            # Add company filter to existing pipeline
                            lookup_pipeline = lookup_stage["pipeline"]
                            # Remove existing company filters and add ours
                            lookup_pipeline = [substage for substage in lookup_pipeline if not (
                                isinstance(substage, dict) and "$match" in substage and "company" in substage.get("$match", {})
                            )]
                            lookup_pipeline.insert(0, {"$match": {"company": user_company}})
                            lookup_stage["pipeline"] = lookup_pipeline

            # Convert ObjectIds
            if collection_name in OBJECTID_FIELDS:
                for i, stage in enumerate(stages):
                    if isinstance(stage, dict):
                        stages[i] = convert_to_objectids(stage, OBJECTID_FIELDS[collection_name])
            
            logger.info(f"Final aggregation stages with MANDATORY company filtering: {json.dumps(stages, default=str, indent=2)}")
            
            # Execute aggregation
            collection = db[collection_name]
            cursor = collection.aggregate(stages)
            results = await cursor.to_list(length=200)
            
            if not results:
                return f"No results found for your company in {collection_name} collection."
            
            return f"Company-filtered aggregation results for {user_company} ({len(results)} items):\n{json.dumps(results, default=str, indent=2)}"
                
        except Exception as e:
            logger.error(f"ERROR in mongodb_aggregation: {str(e)}")
            return f"Aggregation execution failed: {str(e)}"

# User Permissions Tool
class MongoDBGetUserPermissionsInput(BaseModel):
    user_id: Optional[str] = Field(description="The user ID to get permissions for", default=None)

class MongoDBGetUserPermissionsTool(BaseTool):
    name: str = "mongodb_get_user_permissions"
    description: str = "Get current user information and company context"
    args_schema: Type[BaseModel] = MongoDBGetUserPermissionsInput

    def _run(self, user_id: Optional[str] = None) -> str:
        return "This tool only supports async operations"

    async def _arun(self, user_id: Optional[str] = None) -> str:
        try:
            global current_user_context
            
            if not user_id:
                user_id = current_user_context.get("user_id")
            
            if not user_id:
                return "No user context available."
            
            if not isinstance(user_id, str) or not ObjectId.is_valid(user_id):
                return f"Invalid user_id format: {user_id}."
            
            user = await db.users.find_one({"_id": ObjectId(user_id)})
            if not user:
                return f"User with ID {user_id} not found."
            
            company = user.get("company")
            
            result = {
                "user_id": user_id,
                "user_name": user.get("fullName") or user.get("name"),
                "user_email": user.get("email"),
                "account_type": user.get("accountType"),
                "company": str(company) if company else None,
                "status": user.get("status"),
                "permissions": user.get("permissions", {})
            }
            
            return f"User information: {json.dumps(result, default=str, indent=2)}"
            
        except Exception as e:
            return f"Error getting user information: {str(e)}"

# Create tools
tools = [
    MongoDBQueryTool(),
    MongoDBAggregationTool(),
    MongoDBGetUserPermissionsTool(),
]
# Fixed ReAct prompt with ALL required variables
react_prompt = PromptTemplate.from_template("""You are an intelligent MongoDB assistant for the HomeLead Real Estate database.

CORE COLLECTIONS:
- leads: Main customer/prospect data with status tracking
- lead-assignments: Links leads to sales agents  
- properties: Real estate inventory with pricing and availability
- property-bookings: Property reservation records
- brokers: External agent information and commissions
- users: System users and employees
- projects: Real estate development projects

AUTOMATIC CAPABILITIES:
- Company isolation: All queries automatically filtered by user's company
- Field detection: Automatically maps natural language to database fields
- Relationship joins: Automatically adds lookups between related collections
- Security: Users only see their own company's data

QUERY LOGIC:
- Simple data retrieval: Use mongodb_query
- Complex analytics/counting: Use mongodb_aggregation
- User context: Use mongodb_get_user_permissions

FIELD MAPPING:
- "name" maps to leads.name, users.fullName, brokers.name
- "phone" maps to leads.phone, users.phone, brokers.phone  
- "email" maps to leads.email, users.email, brokers.email
- "status" maps to leadStatus, propertyStatus, paymentStatus
- "my" adds assignee filters for current user
- Date terms add createdAt or relevant date filters

You have access to these tools:
{tools}

Available tool names: {tool_names}

Use this exact format:

Question: {input}
Thought: I need to analyze this query and determine the right collection and approach.
Action: [mongodb_query OR mongodb_aggregation OR mongodb_get_user_permissions]
Action Input: {{"collection": "collection_name", "count": true}}
Observation: [Tool result will appear here]
Thought: Based on the results, I can provide the answer.
Final Answer: [Clear response with business context]

Question: {input}
Thought:{agent_scratchpad}""")


# Create agent
agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)

# Global context
current_user_context = {"user_id": None}

# Enhanced agent executor with better user context handling and validation
class CustomAgentExecutor(AgentExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    async def ainvoke(self, inputs, config=None):
        global current_user_context
        user_id = inputs.get("user_id")
        
        # Enhanced validation before setting context
        if not user_id:
            raise ValueError("user_id is required in inputs")
        
        if not isinstance(user_id, str):
            raise ValueError(f"user_id must be a string, got {type(user_id)}: {user_id}")
        
        user_id = user_id.strip()
        if not user_id:
            raise ValueError("user_id cannot be empty")
        
        if not ObjectId.is_valid(user_id):
            raise ValueError(f"Invalid MongoDB ObjectId format: {user_id}")
        
        # Set validated user_id in global context
        current_user_context["user_id"] = user_id
        logger.info(f"User context set for validated user: {user_id}")
        
        try:
            result = await super().ainvoke(inputs, config)
            return result
        except Exception as e:
            logger.error(f"ERROR in agent execution: {str(e)}")
            return {"output": f"I apologize, but I encountered an error processing your request: {str(e)}. Please try rephrasing your question or contact support."}

agent_executor = CustomAgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3,
    early_stopping_method="force"
)

# Export for use in other modules
__all__ = ['agent_executor', 'current_user_context', 'OBJECTID_FIELDS']

logger.info("HomeLead MongoDB Toolkit Agent initialized successfully with enhanced schema intelligence")
