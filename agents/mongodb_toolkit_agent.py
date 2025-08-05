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

# Define ObjectId fields for each collection
OBJECTID_FIELDS = {
    "pages": ["_id"],
    "contact-tags": ["_id", "company"],
    "banks": ["_id", "company"],
    "plans": ["_id"],
    "lands": ["_id", "company", "country"],
    "counters": ["_id"],
    "lead-assignments": ["_id", "company", "lead", "team", "defaultPrimary", "defaultSecondary", "assignee"],
    "general-expenses": ["_id", "company"],
    "contractor-sub-services": ["_id", "contractorService"],
    "property-bookings": ["_id", "company", "lead", "project", "property"],
    "contract-payments": ["_id", "company", "contractor", "project", "contract"],
    "lead-visited-properties": ["_id", "company", "lead", "property"],
    "bhk-types": ["_id"],
    "groups": ["_id", "company"],
    "leads": ["_id", "company", "project", "broker", "bhk", "bhkType"],
    "whatsapp-track": ["_id", "company", "campaign", "target"],
    "project-categories": ["_id"],
    "contractor-services": ["_id"],
    "chats": ["_id", "company", "sender", "group"],
    "campaign-payments": ["_id", "company"],
    "lead-rotations": ["_id", "company", "lead", "team", "assignee"],
    "property-booking-payment-links": ["_id", "company"],
    "agenda-jobs": ["_id"],
    "properties": ["_id", "company", "project", "propertyUnitSubType", "bhk", "bhkType"],
    "property-unit-sub-types": ["_id"],
    "settings": ["_id", "company"],
    "user-sessions": ["_id"],
    "lead-notes": ["_id", "company", "lead", "tag"],
    "projects": ["_id", "company", "land", "category", "country"],
    "chat-cache": ["_id", "group", "company", "sender"],
    "brokers": ["_id", "company", "country"],
    "contracts": ["_id", "company", "contractor", "project"],
    "bank-names": ["_id"],
    "countries": ["_id"],
    "contractors": ["_id", "company", "country"],
    "subscription-payments": ["_id", "company", "plan"],
    "sms-track": ["_id", "company", "campaign", "target"],
    "attendance": ["_id", "company", "user"],
    "teams": ["_id", "company", "teamLead", "defaultPrimary", "defaultSecondary"],
    "documents-and-priorities": ["_id", "company"],
    "broker-payments": ["_id", "company", "project", "property", "booking", "lead", "broker"],
    "rent-payments": ["_id", "company", "project", "property", "tenant"],
    "email-track": ["_id", "company", "campaign", "target"],
    "otps": ["_id", "company"],
    "bhk": ["_id"],
    "contractor-job-types": ["_id"],
    "campaign-templates": ["_id", "company"],
    "sms-balance-requests": ["_id", "company", "campaignPayment"],
    "companies": ["_id", "country", "plan", "superAdmin"],
    "amenities": ["_id"],
    "chat-groups": ["_id", "company"],
    "property-payments": ["_id", "company", "project", "property", "booking"],
    "permissions": ["_id"],
    "users": ["_id", "company", "designation"],
    "miscellaneous-documents": ["_id", "company", "project", "property", "booking"],
    "campaigns": ["_id", "company"],
    "tags": ["_id", "company"],
    "tenants": ["_id", "company", "project", "property"],
    "onboarding-requests": ["_id"],
    "contacts": ["_id", "company"],
    "cold-leads": ["_id", "company"],
    "designations": ["_id"]
}

# Database Schema Knowledge
SCHEMA_KNOWLEDGE = """
Database Schema and Relationships:

COLLECTIONS AND THEIR RELATIONSHIPS:

1. pages (Primary Collection)
   - _id: ObjectId (Primary Key)
   - No foreign keys

2. contact-tags (Primary Collection)
   - _id: ObjectId (Primary Key)
   - company: ObjectId → companies._id (Foreign Key)

3. banks (Primary Collection)
   - _id: ObjectId (Primary Key)
   - company: ObjectId → companies._id (Foreign Key)

4. plans (Primary Collection)
   - _id: ObjectId (Primary Key)
   - No foreign keys

5. lands (Primary Collection)
   - _id: ObjectId (Primary Key)
   - company: ObjectId → companies._id (Foreign Key)
   - country: ObjectId → countries._id (Foreign Key)

6. counters (Primary Collection)
   - _id: ObjectId (Primary Key)
   - No foreign keys

7. lead-assignments (Primary Collection)
   - _id: ObjectId (Primary Key)
   - company: ObjectId → companies._id (Foreign Key)
   - lead: ObjectId → leads._id (Foreign Key)
   - team: ObjectId → teams._id (Foreign Key)
   - defaultPrimary: ObjectId → users._id (Foreign Key)
   - defaultSecondary: ObjectId → users._id (Foreign Key)
   - assignee: ObjectId → users._id (Foreign Key)

8. general-expenses (Primary Collection)
   - _id: ObjectId (Primary Key)
   - company: ObjectId → companies._id (Foreign Key)

9. contractor-sub-services (Primary Collection)
   - _id: ObjectId (Primary Key)
   - contractorService: ObjectId → contractor-services._id (Foreign Key)

10. property-bookings (Primary Collection)
    - _id: ObjectId (Primary Key)
    - company: ObjectId → companies._id (Foreign Key)
    - lead: ObjectId → leads._id (Foreign Key)
    - project: ObjectId → projects._id (Foreign Key)
    - property: ObjectId → properties._id (Foreign Key)

11. contract-payments (Primary Collection)
    - _id: ObjectId (Primary Key)
    - company: ObjectId → companies._id (Foreign Key)
    - contractor: ObjectId → contractors._id (Foreign Key)
    - project: ObjectId → projects._id (Foreign Key)
    - contract: ObjectId → contracts._id (Foreign Key)

12. lead-visited-properties (Primary Collection)
    - _id: ObjectId (Primary Key)
    - company: ObjectId → companies._id (Foreign Key)
    - lead: ObjectId → leads._id (Foreign Key)
    - property: ObjectId → properties._id (Foreign Key)

13. bhk-types (Primary Collection)
    - _id: ObjectId (Primary Key)
    - No foreign keys

14. groups (Primary Collection)
    - _id: ObjectId (Primary Key)
    - company: ObjectId → companies._id (Foreign Key)

15. leads (Primary Collection)
    - _id: ObjectId (Primary Key)
    - company: ObjectId → companies._id (Foreign Key)
    - project: ObjectId → projects._id (Foreign Key)
    - broker: ObjectId → brokers._id (Foreign Key)
    - bhk: ObjectId → bhk._id (Foreign Key)
    - bhkType: ObjectId → bhk-types._id (Foreign Key)

16. whatsapp-track (Primary Collection)
    - _id: ObjectId (Primary Key)
    - company: ObjectId → companies._id (Foreign Key)
    - campaign: ObjectId → campaigns._id (Foreign Key)
    - target: ObjectId → contacts._id (Foreign Key)

17. project-categories (Primary Collection)
    - _id: ObjectId (Primary Key)
    - No foreign keys

18. contractor-services (Primary Collection)
    - _id: ObjectId (Primary Key)
    - No foreign keys

19. chats (Primary Collection)
    - _id: ObjectId (Primary Key)
    - company: ObjectId → companies._id (Foreign Key)
    - sender: ObjectId → users._id (Foreign Key)
    - group: ObjectId → chat-groups._id (Foreign Key)

20. campaign-payments (Primary Collection)
    - _id: ObjectId (Primary Key)
    - company: ObjectId → companies._id (Foreign Key)

21. lead-rotations (Primary Collection)
    - _id: ObjectId (Primary Key)
    - company: ObjectId → companies._id (Foreign Key)
    - lead: ObjectId → leads._id (Foreign Key)
    - team: ObjectId → teams._id (Foreign Key)
    - assignee: ObjectId → users._id (Foreign Key)

22. property-booking-payment-links (Primary Collection)
    - _id: ObjectId (Primary Key)
    - company: ObjectId → companies._id (Foreign Key)

23. agenda-jobs (Primary Collection)
    - _id: ObjectId (Primary Key)
    - No foreign keys

24. properties (Primary Collection)
    - _id: ObjectId (Primary Key)
    - company: ObjectId → companies._id (Foreign Key)
    - project: ObjectId → projects._id (Foreign Key)
    - propertyUnitSubType: ObjectId → property-unit-sub-types._id (Foreign Key)
    - bhk: ObjectId → bhk._id (Foreign Key)
    - bhkType: ObjectId → bhk-types._id (Foreign Key)

25. property-unit-sub-types (Primary Collection)
    - _id: ObjectId (Primary Key)
    - No foreign keys

26. settings (Primary Collection)
    - _id: ObjectId (Primary Key)
    - company: ObjectId → companies._id (Foreign Key)

27. user-sessions (Primary Collection)
    - _id: ObjectId (Primary Key)
    - No foreign keys

28. lead-notes (Primary Collection)
    - _id: ObjectId (Primary Key)
    - company: ObjectId → companies._id (Foreign Key)
    - lead: ObjectId → leads._id (Foreign Key)
    - tag: ObjectId → tags._id (Foreign Key)

29. projects (Primary Collection)
    - _id: ObjectId (Primary Key)
    - company: ObjectId → companies._id (Foreign Key)
    - land: ObjectId → lands._id (Foreign Key)
    - category: ObjectId → project-categories._id (Foreign Key)
    - country: ObjectId → countries._id (Foreign Key)

30. chat-cache (Primary Collection)
    - _id: ObjectId (Primary Key)
    - group: ObjectId → chat-groups._id (Foreign Key)
    - company: ObjectId → companies._id (Foreign Key)
    - sender: ObjectId → users._id (Foreign Key)

31. brokers (Primary Collection)
    - _id: ObjectId (Primary Key)
    - company: ObjectId → companies._id (Foreign Key)
    - country: ObjectId → countries._id (Foreign Key)

32. contracts (Primary Collection)
    - _id: ObjectId (Primary Key)
    - company: ObjectId → companies._id (Foreign Key)
    - contractor: ObjectId → contractors._id (Foreign Key)
    - project: ObjectId → projects._id (Foreign Key)

33. bank-names (Primary Collection)
    - _id: ObjectId (Primary Key)
    - No foreign keys

34. countries (Primary Collection)
    - _id: ObjectId (Primary Key)
    - No foreign keys

35. contractors (Primary Collection)
    - _id: ObjectId (Primary Key)
    - company: ObjectId → companies._id (Foreign Key)
    - country: ObjectId → countries._id (Foreign Key)

36. subscription-payments (Primary Collection)
    - _id: ObjectId (Primary Key)
    - company: ObjectId → companies._id (Foreign Key)
    - plan: ObjectId → plans._id (Foreign Key)

37. sms-track (Primary Collection)
    - _id: ObjectId (Primary Key)
    - company: ObjectId → companies._id (Foreign Key)
    - campaign: ObjectId → campaigns._id (Foreign Key)
    - target: ObjectId → contacts._id (Foreign Key)

38. attendance (Primary Collection)
    - _id: ObjectId (Primary Key)
    - company: ObjectId → companies._id (Foreign Key)
    - user: ObjectId → users._id (Foreign Key)

39. teams (Primary Collection)
    - _id: ObjectId (Primary Key)
    - company: ObjectId → companies._id (Foreign Key)
    - teamLead: ObjectId → users._id (Foreign Key)
    - defaultPrimary: ObjectId → users._id (Foreign Key)
    - defaultSecondary: ObjectId → users._id (Foreign Key)

40. documents-and-priorities (Primary Collection)
    - _id: ObjectId (Primary Key)
    - company: ObjectId → companies._id (Foreign Key)

41. broker-payments (Primary Collection)
    - _id: ObjectId (Primary Key)
    - company: ObjectId → companies._id (Foreign Key)
    - project: ObjectId → projects._id (Foreign Key)
    - property: ObjectId → properties._id (Foreign Key)
    - booking: ObjectId → property-bookings._id (Foreign Key)
    - lead: ObjectId → leads._id (Foreign Key)
    - broker: ObjectId → brokers._id (Foreign Key)

42. rent-payments (Primary Collection)
    - _id: ObjectId (Primary Key)
    - company: ObjectId → companies._id (Foreign Key)
    - project: ObjectId → projects._id (Foreign Key)
    - property: ObjectId → properties._id (Foreign Key)
    - tenant: ObjectId → tenants._id (Foreign Key)

43. email-track (Primary Collection)
    - _id: ObjectId (Primary Key)
    - company: ObjectId → companies._id (Foreign Key)
    - campaign: ObjectId → campaigns._id (Foreign Key)
    - target: ObjectId → contacts._id (Foreign Key)

44. otps (Primary Collection)
    - _id: ObjectId (Primary Key)
    - company: ObjectId → companies._id (Foreign Key)

45. bhk (Primary Collection)
    - _id: ObjectId (Primary Key)
    - No foreign keys

46. contractor-job-types (Primary Collection)
    - _id: ObjectId (Primary Key)
    - No foreign keys

47. campaign-templates (Primary Collection)
    - _id: ObjectId (Primary Key)
    - company: ObjectId → companies._id (Foreign Key)

48. sms-balance-requests (Primary Collection)
    - _id: ObjectId (Primary Key)
    - company: ObjectId → companies._id (Foreign Key)
    - campaignPayment: ObjectId → campaign-payments._id (Foreign Key)

49. companies (Primary Collection)
    - _id: ObjectId (Primary Key)
    - country: ObjectId → countries._id (Foreign Key)
    - plan: ObjectId → plans._id (Foreign Key)
    - superAdmin: ObjectId → users._id (Foreign Key)

50. amenities (Primary Collection)
    - _id: ObjectId (Primary Key)
    - No foreign keys

51. chat-groups (Primary Collection)
    - _id: ObjectId (Primary Key)
    - company: ObjectId → companies._id (Foreign Key)

52. property-payments (Primary Collection)
    - _id: ObjectId (Primary Key)
    - company: ObjectId → companies._id (Foreign Key)
    - project: ObjectId → projects._id (Foreign Key)
    - property: ObjectId → properties._id (Foreign Key)
    - booking: ObjectId → property-bookings._id (Foreign Key)

53. permissions (Primary Collection)
    - _id: ObjectId (Primary Key)
    - No foreign keys

54. users (Primary Collection)
    - _id: ObjectId (Primary Key)
    - company: ObjectId → companies._id (Foreign Key)
    - designation: ObjectId → designations._id (Foreign Key)

55. miscellaneous-documents (Primary Collection)
    - _id: ObjectId (Primary Key)
    - company: ObjectId → companies._id (Foreign Key)
    - project: ObjectId → projects._id (Foreign Key)
    - property: ObjectId → properties._id (Foreign Key)
    - booking: ObjectId → property-bookings._id (Foreign Key)

56. campaigns (Primary Collection)
    - _id: ObjectId (Primary Key)
    - company: ObjectId → companies._id (Foreign Key)

57. tags (Primary Collection)
    - _id: ObjectId (Primary Key)
    - company: ObjectId → companies._id (Foreign Key)

58. tenants (Primary Collection)
    - _id: ObjectId (Primary Key)
    - company: ObjectId → companies._id (Foreign Key)
    - project: ObjectId → projects._id (Foreign Key)
    - property: ObjectId → properties._id (Foreign Key)

59. onboarding-requests (Primary Collection)
    - _id: ObjectId (Primary Key)
    - No foreign keys

60. contacts (Primary Collection)
    - _id: ObjectId (Primary Key)
    - company: ObjectId → companies._id (Foreign Key)

61. cold-leads (Primary Collection)
    - _id: ObjectId (Primary Key)
    - company: ObjectId → companies._id (Foreign Key)

62. designations (Primary Collection)
    - _id: ObjectId (Primary Key)
    - No foreign keys

USER-BASED FILTERING:
- Users can only see data from their own company
- Super admins can see all data
- Regular users are filtered by company field
- Use user_id to get user permissions and company access

AGGREGATION EXAMPLES:
- "Top performing team": Group leads by assignedTo → users.team, count "Closed - Won" status
- "Total rent by tenant": Group rent-payments by tenant, sum amounts
- "Lead conversion rate": Count leads by status, calculate percentages
- "Revenue by property": Group rent-payments by property, sum amounts
- "Broker performance": Group broker-payments by broker, sum amounts
- "Project revenue": Group property-payments by project, sum amounts

QUERY EXAMPLES:
- "Show all leads for company X": Query leads where company = company_id
- "Find users who are super admins": Query companies where superAdmin exists, then get user details
- "Get total rent paid by tenant": Use aggregation to group and sum
- "Show properties for a specific project": Query properties where project = project_id
- "Find all bookings for a lead": Query property-bookings where lead = lead_id
- "Get all payments for a property": Query property-payments where property = property_id
"""

# Custom MongoDB Tools
class MongoDBQueryInput(BaseModel):
    query: str = Field(description="The MongoDB query in JSON format")
    user_id: Optional[str] = Field(description="User ID for permission-based filtering", default=None)
    collection: Optional[str] = Field(description="Collection name", default=None)
    count: Optional[bool] = Field(description="Whether to count documents", default=False)
    projection: Optional[str] = Field(description="Projection fields", default=None)
    limit: Optional[int] = Field(description="Limit number of results", default=None)

class MongoDBQueryTool(BaseTool):
    name: str = "mongodb_query"
    description: str = "Execute simple MongoDB queries for basic lookups. For complex queries with grouping, counting, or joining collections, use mongodb_aggregation tool instead. This tool automatically filters results based on user permissions."
    args_schema: Type[BaseModel] = MongoDBQueryInput

    def _run(self, query: str, user_id: Optional[str] = None) -> str:
        # Synchronous version - not used but required
        return "This tool only supports async operations"

    async def _arun(self, query: str, user_id: Optional[str] = None, collection: Optional[str] = None, count: Optional[bool] = False, projection: Optional[str] = None, limit: Optional[int] = None) -> str:
        try:
            # Get user_id from global context if not provided
            global current_user_context
            if not user_id:
                user_id = current_user_context.get("user_id")
            
            # Parse the JSON query if it's a string, otherwise use the direct parameters
            if isinstance(query, str):
                try:
                    query_data = json.loads(query)
                    collection_name = query_data.get("collection") or collection
                    mongo_query = query_data.get("query", {})
                    limit = query_data.get("limit") or limit
                    count = query_data.get("count", False) or count
                    projection = query_data.get("projection") or projection
                    user_id = query_data.get("user_id") or user_id
                except json.JSONDecodeError:
                    # If not JSON, treat as collection name
                    collection_name = query
                    mongo_query = {}
            else:
                collection_name = collection
                mongo_query = {}
            
            if not collection_name:
                return "Error: collection name is required"
            
            collection = db[collection_name]
            
            # Convert string values to ObjectIds for this collection
            if collection_name in OBJECTID_FIELDS:
                mongo_query = convert_to_objectids(mongo_query, OBJECTID_FIELDS[collection_name])
            
            # Apply user-based filtering if user_id is provided
            # Note: user_id should be passed from the agent executor context
            if user_id:
                try:
                    # Get user details and permissions - use _id for users collection
                    user = await db.users.find_one({"_id": ObjectId(user_id)})
                    if not user:
                        return f"Error: User with ID {user_id} not found"
                    
                    user_company = user.get("company")
                    user_permissions = user.get("permissions", [])
                    
                    # Check if user is super admin - use _id for companies collection
                    is_super_admin = await db.companies.find_one({"superAdmin": ObjectId(user_id)})
                    
                    # Apply user-based filtering for non-super admins
                    if not is_super_admin and user_company:
                        # Add appropriate filter based on collection
                        if collection_name == "users":
                            # For users collection, filter by current user ID
                            if "_id" in mongo_query:
                                if mongo_query["_id"] != user_id:
                                    return "Error: You can only access your own user data"
                            else:
                                mongo_query["_id"] = ObjectId(user_id)
                        elif collection_name in ["leads", "lead-assignments"]:
                            # For leads, filter by assignedTo field
                            if "assignedTo" in mongo_query:
                                if mongo_query["assignedTo"] != ObjectId(user_id):
                                    return "Error: You can only access leads assigned to you"
                            else:
                                mongo_query["assignedTo"] = ObjectId(user_id)
                        else:
                            # For other collections, filter by company
                            if "company" in mongo_query:
                                if mongo_query["company"] != user_company:
                                    return "Error: You can only access data from your own company"
                            else:
                                mongo_query["company"] = user_company
                    
                    # Apply permission-based filtering
                    if not is_super_admin:
                        # Check if user has permission to access this collection
                        # Common permission patterns: "leads[]", "users[]", "companies[]", etc.
                        collection_permission = f"{collection_name}[]"
                        read_permission = f"{collection_name}.read"
                        write_permission = f"{collection_name}.write"
                        
                        # Check for various permission formats
                        has_permission = (
                            collection_permission in user_permissions or
                            read_permission in user_permissions or
                            write_permission in user_permissions or
                            "admin" in user_permissions or
                            "super_admin" in user_permissions
                        )
                        
                        if not has_permission:
                            return f"Error: You don't have permission to access {collection_name}. Required permissions: {collection_permission}, {read_permission}, or {write_permission}. Your permissions: {user_permissions}"
                            
                except Exception as e:
                    return f"Error applying user filtering: {str(e)}"
            
            if count:
                # Count documents
                result = await collection.count_documents(mongo_query)
                return f"Total count: {result}"
            else:
                # Find documents with smart projection
                if not projection:
                    # Default projection to show only relevant fields
                    if collection_name == "leads":
                        projection = {"name": 1, "email": 1, "phone": 1, "status": 1, "_id": 0}
                    elif collection_name == "users":
                        projection = {"name": 1, "email": 1, "_id": 0}
                    elif collection_name == "companies":
                        projection = {"name": 1, "clientName": 1, "_id": 0}
                    elif collection_name == "properties":
                        projection = {"name": 1, "_id": 0}
                    elif collection_name == "rent-payments":
                        projection = {"amount": 1, "_id": 0}
                    else:
                        projection = {"name": 1, "_id": 0}  # Default to name field
                
                cursor = collection.find(mongo_query, projection)
                if limit:
                    cursor = cursor.limit(limit)
                
                documents = await cursor.to_list(length=limit or 50)  # Reduced default limit
                
                if not documents:
                    return f"No {collection_name} found matching the criteria."
                
                # Format results in a user-friendly way
                if len(documents) == 1:
                    return f"Found 1 {collection_name[:-1]}: {json.dumps(documents[0], default=str, indent=2)}"
                else:
                    return f"Found {len(documents)} {collection_name}: {json.dumps(documents, default=str, indent=2)}"
                
        except json.JSONDecodeError:
            return "Error: Invalid JSON format"
        except Exception as e:
            return f"Error executing query: {str(e)}"

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
        # Synchronous version - not used but required
        return "This tool only supports async operations"

    async def _arun(self, pipeline: str, user_id: Optional[str] = None, collection: Optional[str] = None, stages: Optional[str] = None) -> str:
        try:
            # Get user_id from global context if not provided
            global current_user_context
            if not user_id:
                user_id = current_user_context.get("user_id")
            
            # Parse the aggregation pipeline if it's a string, otherwise use direct parameters
            if isinstance(pipeline, str):
                try:
                    pipeline_data = json.loads(pipeline)
                    collection_name = pipeline_data.get("collection") or collection
                    stages = pipeline_data.get("stages", []) or stages
                    user_id = pipeline_data.get("user_id") or user_id
                except json.JSONDecodeError:
                    # If not JSON, treat as collection name
                    collection_name = pipeline
                    stages = []
            else:
                collection_name = collection
                stages = stages or []
            
            if not collection_name:
                return "Error: collection name is required"
            
            collection = db[collection_name]
            
            # Convert ObjectIds in aggregation stages
            if collection_name in OBJECTID_FIELDS:
                for stage in stages:
                    if isinstance(stage, dict):
                        for stage_type, stage_data in stage.items():
                            if isinstance(stage_data, dict):
                                stage[stage_type] = convert_to_objectids(stage_data, OBJECTID_FIELDS[collection_name])
            
            # Apply user-based filtering if user_id is provided
            if user_id:
                try:
                    # Get user details and permissions - use _id for users collection
                    user = await db.users.find_one({"_id": ObjectId(user_id)})
                    if not user:
                        return f"Error: User with ID {user_id} not found"
                    
                    user_company = user.get("company")
                    user_permissions = user.get("permissions", [])
                    
                    # Check if user is super admin - use _id for companies collection
                    is_super_admin = await db.companies.find_one({"superAdmin": ObjectId(user_id)})
                    
                    # Apply company-based filtering for non-super admins
                    if not is_super_admin and user_company:
                        # Replace placeholders with actual user and company IDs
                        def replace_placeholders(stages_list):
                            for stage in stages_list:
                                if isinstance(stage, dict):
                                    for stage_type, stage_data in stage.items():
                                        if isinstance(stage_data, dict):
                                            # Replace CURRENT_USER_ID in $match stages
                                            if stage_type == "$match":
                                                for field, value in stage_data.items():
                                                    if value == "CURRENT_USER_ID":
                                                        stage_data[field] = ObjectId(user_id)
                                                    elif value == "USER_COMPANY_ID":
                                                        stage_data[field] = user_company
                                            # Recursively check nested stages
                                            replace_placeholders([stage_data])
                        
                        replace_placeholders(stages)
                        
                        # Add user-specific filters if not already present
                        has_user_filter = any(
                            isinstance(stage, dict) and "$match" in stage and 
                            any(field in stage.get("$match", {}) for field in ["_id", "assignedTo", "assignee", "company"])
                            for stage in stages
                        )
                        
                        if not has_user_filter:
                            # Add appropriate filter based on collection
                            if collection_name == "users":
                                user_filter = {"$match": {"_id": ObjectId(user_id)}}
                            elif collection_name in ["leads", "lead-assignments"]:
                                user_filter = {"$match": {"assignedTo": ObjectId(user_id)}}
                            else:
                                user_filter = {"$match": {"company": user_company}}
                            stages.insert(0, user_filter)
                    
                    # Apply permission-based filtering
                    if not is_super_admin:
                        # Check if user has permission to access this collection
                        # Common permission patterns: "leads[]", "users[]", "companies[]", etc.
                        collection_permission = f"{collection_name}[]"
                        read_permission = f"{collection_name}.read"
                        write_permission = f"{collection_name}.write"
                        
                        # Check for various permission formats
                        has_permission = (
                            collection_permission in user_permissions or
                            read_permission in user_permissions or
                            write_permission in user_permissions or
                            "admin" in user_permissions or
                            "super_admin" in user_permissions
                        )
                        
                        if not has_permission:
                            return f"Error: You don't have permission to access {collection_name}. Required permissions: {collection_permission}, {read_permission}, or {write_permission}. Your permissions: {user_permissions}"
                            
                except Exception as e:
                    return f"Error applying user filtering: {str(e)}"
            
            # Execute aggregation pipeline
            cursor = collection.aggregate(stages)
            results = await cursor.to_list(length=100)  # Limit aggregation results
            
            if not results:
                return f"No results found for aggregation on {collection_name}."
            
            # Format results in a user-friendly way
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
    description: str = "Get user permissions and company access for filtering queries"
    args_schema: Type[BaseModel] = MongoDBGetUserPermissionsInput

    def _run(self, user_id: str) -> str:
        # Synchronous version - not used but required
        return "This tool only supports async operations"

    async def _arun(self, user_id: str) -> str:
        try:
            # Get user_id from global context if not provided
            global current_user_context
            if not user_id:
                user_id = current_user_context.get("user_id")
                if not user_id:
                    return "Error: No user_id provided"
            
            # Get user details
            user = await db.users.find_one({"_id": ObjectId(user_id)})
            if not user:
                return f"Error: User with ID {user_id} not found"
            
            # Get user's company details
            user_company = user.get("company")
            company_details = None
            if user_company:
                company_details = await db.companies.find_one({"_id": user_company})
            
            # Check if user is super admin
            is_super_admin = await db.companies.find_one({"superAdmin": ObjectId(user_id)})
            
            result = {
                "user_id": user_id,
                "user_name": user.get("name"),
                "user_email": user.get("email"),
                "company_id": str(user_company) if user_company else None,
                "company_name": company_details.get("name") if company_details else None,
                "is_super_admin": bool(is_super_admin),
                "permissions": user.get("permissions", [])
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
        # Synchronous version - not used but required
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
        # Synchronous version - not used but required
        return "This tool only supports async operations"

    async def _arun(self, collection: str) -> str:
        try:
            coll = db[collection]
            # Get one document to understand the schema
            sample = await coll.find_one()
            if sample:
                # Show only key fields for better readability
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
        # Synchronous version - not used but required
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
react_prompt = PromptTemplate.from_template("""You are a helpful AI assistant that can answer questions about the MongoDB database with user-based filtering and aggregation support. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT GUIDELINES:
1. ALWAYS use mongodb_aggregation tool for complex queries (grouping, counting, joining collections)
2. Use mongodb_query tool only for simple lookups
3. ALWAYS include relevant fields from related collections using $lookup
4. Filter results by CURRENT USER's ID or company using foreign key relationships
5. For user-specific data, use $match with current user's ID or company ID
6. Include meaningful fields like name, email, status, amount, etc. in results
7. Use $project to show only relevant fields for better user experience
8. For relationships, use proper ObjectId field names from schema
9. Always apply user-based filtering for security (current user's data only)

AGGREGATION FORMATS:
- Current user's data: {{"collection": "users", "stages": [{{"$match": {{"_id": "CURRENT_USER_ID"}}}}, {{"$project": {{"name": 1, "email": 1, "company": 1}}}}]}}
- User's company data: {{"collection": "tenants", "stages": [{{"$match": {{"company": "USER_COMPANY_ID"}}}}, {{"$count": "total"}}]}}
- User's assigned leads: {{"collection": "leads", "stages": [{{"$match": {{"assignedTo": "CURRENT_USER_ID"}}}}, {{"$lookup": {{"from": "users", "localField": "assignedTo", "foreignField": "_id", "as": "assignee"}}}}, {{"$project": {{"name": 1, "status": 1, "assignee_name": "$assignee.name"}}}}]}}
- User's company payments: {{"collection": "rent-payments", "stages": [{{"$match": {{"company": "USER_COMPANY_ID"}}}}, {{"$lookup": {{"from": "tenants", "localField": "tenant", "foreignField": "_id", "as": "tenant_info"}}}}, {{"$project": {{"amount": 1, "tenant_name": "$tenant_info.name"}}}}]}}
- Note: CURRENT_USER_ID and USER_COMPANY_ID are automatically replaced

OBJECTID FIELDS BY COLLECTION (Key Collections):
- leads: _id, company, project, broker, bhk, bhkType
- users: _id, company, designation
- companies: _id, country, plan, superAdmin
- teams: _id, company, teamLead, defaultPrimary, defaultSecondary
- properties: _id, company, project, propertyUnitSubType, bhk, bhkType
- projects: _id, company, land, category, country
- rent-payments: _id, company, project, property, tenant
- tenants: _id, company, project, property
- brokers: _id, company, country
- contractors: _id, company, country
- campaigns: _id, company
- lead-assignments: _id, company, lead, team, defaultPrimary, defaultSecondary, assignee
- property-bookings: _id, company, lead, project, property
- property-payments: _id, company, project, property, booking
- broker-payments: _id, company, project, property, booking, lead, broker
- contract-payments: _id, company, contractor, project, contract
- lead-notes: _id, company, lead, tag
- chats: _id, company, sender, group
- attendance: _id, company, user

AGGREGATION EXAMPLES:
- Current user's name: Match by current user ID, project name field
- User's assigned leads: Match by current user ID in assignedTo field
- User's company tenants: Match by user's company ID
- User's company payments: Match by company, lookup tenant details
- User's lead assignments: Match by current user ID in assignee field
- User's company properties: Match by company, include project details

USER-BASED FILTERING:
- ALWAYS use aggregation pipelines for complex queries
- Filter results by CURRENT USER's ID or company using foreign key relationships
- Include relevant fields from related collections using $lookup
- Use $project to show meaningful fields (name, email, status, amount, etc.)
- Replace CURRENT_USER_ID and USER_COMPANY_ID placeholders automatically
- For relationships, use proper ObjectId field names from schema
- Always apply user-based filtering for security (current user's data only)
- Include fields like: name, email, phone, status, amount, clientName, etc.
- For user-specific queries, match by current user's ID
- For company-wide queries, match by user's company ID

Question: {input}
{agent_scratchpad}""")

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
        current_user_context["user_id"] = inputs.get("user_id")
        return await super().ainvoke(inputs, config)

agent_executor = CustomAgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)
