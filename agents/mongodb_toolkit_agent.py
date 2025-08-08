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
    "banks": ["_id", "company", "bankNameId._id"],
    "plans": ["_id"],
    "lands": [
        "_id", "company", "country",
        "owners[]->_id", "pastOwners[]->_id"
    ],
    "counters": ["_id"],
    "lead-assignments": [
        "_id", "assignee", "company", "defaultPrimary", "defaultSecondary", "lead", "team"
    ],
    "general-expenses": ["_id", "company"],
    "contractor-sub-services": ["_id", "contractorService"],
    "property-bookings": [
        "_id", "company", "lead", "project", "property",
        "clpPhases[]->_id", "documentReferences[]->_id", "extraCharges[]->_id", "timelineDocuments[]->_id"
    ],
    "contract-payments": ["_id", "company", "contract", "contractor", "project"],
    "lead-visited-properties": ["_id", "company", "lead", "property"],
    "bhk-types": ["_id"],
    "groups": ["_id", "company"],
    "leads": ["_id", "bhk", "bhkType", "broker", "company", "project"],
    "whatsapp-track": ["_id", "campaign", "company", "target"],
    "project-categories": ["_id"],
    "contractor-services": ["_id"],
    "chats": ["_id", "company", "group", "sender", "actions[]->_id"],
    "campaign-payments": ["_id", "company"],
    "lead-rotations": ["_id", "assignee", "company", "lead", "team"],
    "property-booking-payment-links": ["_id", "company"],
    "agenda-jobs": ["_id", "data.campaign", "data.company"],
    "properties": ["_id", "bhk", "bhkType", "company", "project", "propertyUnitSubType"],
    "property-unit-sub-types": ["_id"],
    "settings": [
        "_id", "company", "attendance.shiftTimingDayWise[]->_id", "credentials.easeBuzz.credentials[]->_id", "general.currency[]->_id"
    ],
    "user-sessions": ["_id"],
    "lead-notes": ["_id", "company", "lead", "tag"],
    "projects": [
        "_id", "company", "land", "category", "country",
        "blocksResidential[]->_id", "blocksResidential[]->floors[]->_id", "clpPhases[]->_id", "nearByLocations[]->_id",
        "totalResidential.blocks[]->_id", "totalResidential.blocks[]->floors[]->_id", "totalResidential.blocks[]->floors[]->units[]->_id"
    ],
    "chat-cache": ["_id", "group", "company", "sender", "actions[]->_id"],
    "brokers": ["_id", "company", "country", "bankDetails.bankNameId._id", "documentReferences[]->_id"],
    "contracts": [
        "_id", "company", "contractor", "project",
        "milestones[]->_id", "serviceTypes[]->_id", "witnesses[]->_id"
    ],
    "bank-names": ["_id"],
    "countries": ["_id", "states[]->_id", "states[]->cities[]->_id"],
    "contractors": [
        "_id", "company", "country", "contactPersons[]->_id", "documents[]->_id", "serviceTypes[]->_id"
    ],
    "subscription-payments": ["_id", "company", "plan"],
    "sms-track": ["_id", "campaign", "company", "target"],
    "attendance": ["_id", "company", "user", "logs[]->_id"],
    "teams": ["_id", "company", "defaultPrimary", "defaultSecondary", "teamLead"],
    "documents-and-priorities": ["_id", "company"],
    "broker-payments": ["_id", "booking", "broker", "company", "lead", "project", "property"],
    "rent-payments": ["_id", "company", "project", "property", "tenant"],
    "email-track": ["_id", "campaign", "company", "target"],
    "otps": ["_id", "company"],
    "bhk": ["_id"],
    "contractor-job-types": ["_id"],
    "campaign-templates": ["_id", "company", "variables[]->_id"],
    "sms-balance-requests": ["_id", "campaignPayment", "company"],
    "companies": ["_id", "country", "plan", "superAdmin"],
    "amenities": ["_id"],
    "chat-groups": ["_id", "company", "members[]->_id", "members[]->user"],
    "property-payments": ["_id", "booking", "company", "project", "property"],
    "permissions": ["_id"],
    "users": ["_id", "company", "designation", "groups[]"],
    "miscellaneous-documents": ["_id", "booking", "company", "project", "property"],
    "campaigns": ["_id", "company"],
    "tags": ["_id", "company"],
    "tenants": ["_id", "company", "project", "property", "documentReferences[]->_id", "members[]->_id"],
    "onboarding-requests": ["_id", "country", "plan"],
    "contacts": ["_id", "company", "tags[]"],
    "cold-leads": ["_id", "company"],
    "designations": ["_id"]
}

# Database Schema Knowledge
SCHEMA_KNOWLEDGE = """SCHEMAS: Dict[str, Dict[str, Any]] = {
"companies":{"fields":{"_id":"id","clientName":"str","name":"str","contactPersonName":"str","subDomain":"str","primaryPhone":"str","secondaryPhone":"str","primaryEmail":"str","contactPersonEmail":"str","contactPersonPhone":"str","address":"str","country":"id","state":"id","city":"id","addOnServices":"array<enum:Sms|Email|IVR>","plan":"id","planStartDate":"datetime","planEndDate":"datetime","maxNoOfUsers":"int","maxNoOfProjects":"int","amountPaid":"number","totalGeneralExpenses":"number","allowPlanRenewalOnSamePrice":"bool","smartPingBalance":"int","createdAt":"datetime"}},
"contracts":{"fields":{"_id":"id","amount":"number","anyReasonableExpenses":"number","chargesOnLatePayment":"number","company":"id","contractRate":"number","contractRateType":"enum:Hourly|Weekly","contractor":"id","startDate":"datetime","endDate":"datetime","generatedReferenceNo":"str","isSubContractAllowed":"bool","milestones":"array<object>","payForEachInvoiceDue":"bool","project":"id","scopeOfWork":"str","serviceTypes":"array<str>","terminationPeriod":"number","witnesses":"array<object>"}},
"contractors":{"fields":{"_id":"id","company":"id","name":"str","countryCode":"str","phone":"str","email":"str","address":"str","country":"id","state":"id","city":"id","zipCode":"str","contractorType":"enum:Firm|Individual","contractorRateType":"enum:Hourly|Weekly","suitableFor":"enum:Residential|Commercial","isGoodsSupplier":"bool","yearsOfExperience":"int","createdAt":"datetime","updatedAt":"datetime"}},
"contract-payments":{"fields":{"_id":"id","company":"id","contractor":"id","project":"id","contract":"id","ordinal":"int","amount":"number","paymentStatus":"enum:Paid|Due","paymentMode":"enum:Cash|Cheque|Online","generatedReferenceNo":"str","referenceNo":"str","dueDate":"datetime","receivedOn":"datetime","createdAt":"datetime","updatedAt":"datetime"}},
"brokers":{"fields":{"_id":"id","company":"id","name":"str","phone":"str","country":"id","city":"id","state":"id","address":"str","zipCode":"str","documentReferences":"array<object<title:str,referenceNo:str,_id:id>>","commissionPercent":"number","realEstateLicenseDetails":"object<licenseNo:str,licenseIssueDate:datetime,licenseExpirationDate:datetime>","yearStartedInRealEstate":"int","createdAt":"datetime","updatedAt":"datetime"}},
"broker-payments":{"fields":{"_id":"id","company":"id","project":"id","property":"id","booking":"id","lead":"id","broker":"id","amount":"number","paymentStatus":"enum:Paid|Due","paymentMode":"enum:Cash|Online|Cheque","generatedReferenceNo":"str","paidOn":"datetime","createdAt":"datetime","updatedAt":"datetime"}},
"cold-leads":{"fields":{"_id":"id","company":"id","countryCode":"str","phone":"str","coldLeadStatus":"enum:Pending|Converted","createdAt":"datetime","updatedAt":"datetime"}},
"general-expenses":{"fields":{"_id":"id","company":"id","payeeName":"str","expenseType":"enum:Against Salary","amount":"number","taxes":"number","payableAmount":"number","paymentMode":"enum:Cash|Cheque|Online","generatedReferenceNo":"str","referenceNo":"str","dateOfPayment":"datetime","createdAt":"datetime","updatedAt":"datetime"}},
"lands":{"fields":{"_id":"id","company":"id","name":"str","propertyType":"array<enum:Residential|Commercial>","address":"str","country":"id","state":"id","city":"id","zipCode":"str","location":"object","purchasePrice":"number","currentMarketValue":"number","isOnLease":"bool","rentalIncome":"number","occupancyStatus":"enum:Vacant|Occupied","nearByArea":"str","isAnyConstructionExists":"bool","plotSize":"number","sizeType":"enum:Square Meters|Hectares|Gaj|Square Feet|Square Yards|Acres","amenities":"array<id>","owners":"array<object>","createdAt":"datetime","updatedAt":"datetime"}},
"lead-assignments":{"fields":{"_id":"id","company":"id","lead":"id","team":"id","defaultPrimary":"id","defaultSecondary":"id","assignee":"id","createdAt":"datetime","updatedAt":"datetime"}},
"lead-notes":{"fields":{"_id":"id","company":"id","lead":"id","communicationType":"enum:Call|In Person|WhatsApp","meetingDateTime":"datetime","siteVisitStatus":"enum:Pending|Scheduled|Visited","siteVisitScheduledDate":"datetime","nextSiteVisitScheduledDate":"datetime","callDuration":"number","receiverPhone":"str","virtualPhone":"str","callDate":"datetime","createdAt":"datetime"}},
"lead-rotations":{"fields":{"_id":"id","company":"id","lead":"id","team":"id","assignee":"id","date":"datetime","createdAt":"datetime","updatedAt":"datetime"}},
"leads": {"fields": {"_id":"id","broker":"id","buyingTimeline":["0 TO 3 months","0 TO 6 months"],"commissionPercent":3,"company":"id","createdAt":["2025-03-31T06:55:42.351Z"],"email":["is@mail.com"],"leadStatus":["On going","Converted","Temporary Converted"],"maxBudget":1000,"minBudget":1000,"name":["Ishaan"],"phone":["9883726473"],"project":"id","propertyType":["Residential","Commercial"],"rotationCount":9,"secondaryPhone":["7689001334"],"sourceType":["Direct","Broker","Housing.com","99acres.com","MagicBricks.com"],"updatedAt": ["2025-03-31T11:38:15.076Z"],"bhk":"id","bhkType":"id"}},
"lead-visited-properties":{"fields":{"_id":"id","company":"id","lead":"id","property":"id","createdAt":"datetime","updatedAt":"datetime"}},
"plans":{"fields":{"_id":"id","name":"str","paymentDuration":"enum:Monthly|Quarterly|Yearly","price":"number","description":"str","features":"array<str>","maxNoOfUsers":"int","maxNoOfProjects":"int","priceInUSD":"number"}},
"projects":{"fields":{"_id":"id","company":"id","land":"id","name":"str","projectType":"array<enum:Residential|Commercial>","projectUnitSubType":"array<id>","category":"id","projectStatus":"enum:Under construction","minBudget":"number","maxBudget":"number","startDate":"datetime","completionDate":"datetime","address":"str","country":"id","state":"id","city":"id","zipCode":"str","location":"object","reraRegistrationNumber":"str","projectRegistrationNumber":"str","isGovtApproved":"bool","govtApprovedDocuments":"array<str>","noOfUnitsResidential":"int","noOfBlocksResidential":"int","blocksResidential":"array<object>"}},
"properties":{"fields":{"_id":"id","company":"id","project":"id","propertyType":"enum:Residential|Commercial","propertyUnitSubType":"id","bhk":"id","bhkType":"id","blockName":"str","floorName":"str","series":"str","flatNo":"int","furnishedStatus":"enum:Furnished|Unfurnished|Semi-furnished","minBudget":"number","maxBudget":"number","facing":"enum:North|East|West|South","vastuCompliant":"bool","propertyArea":"number","carpetArea":"number","builtUpArea":"number","superBuiltUpArea":"number","propertyAreaType":"enum:Square Feet|Square Meters","noOfBalconies":"int","noOfBathRooms":"int","noOfBedRooms":"int","noOfKitchens":"int","noOfDrawingRooms":"int","noOfParkingLots":"int","propertyStatus":"enum:Available|Sold|Pending","createdAt":"datetime","updatedAt":"datetime"}},
"property-bookings":{"fields":{"_id":"id","company":"id","lead":"id","project":"id","property":"id","bookingDate":"datetime","bookingType":"enum:Temporary Booking|Final Booking","saleablePrice":"number","basicPrice":"number","bookingAmount":"number","taxPercent":"number","taxAmount":"number","bookingPaymentStatus":"enum:Paid|Due|Refunded","paymentTerm":"enum:Loan|Without Loan","financeDeptStatus":"enum:Pending|Accepted|Rejected","extraCharges":"array<object<chargeFor:str,amount:number,_id:id>>","createdAt":"datetime","updatedAt":"datetime"}},
"property-payments":{"fields":{"_id":"id","company":"id","project":"id","property":"id","booking":"id","paymentType":"enum:Booking Amount|CLP","transactionType":"enum:Credit|Debit","amount":"number","paymentStatus":"enum:Paid|Due","paymentMode":"enum:Cash|Cheque|Online","receivedOn":"datetime","generatedReferenceNo":"str","referenceNo":"str","dueFrom":"str","clpPhase":"str","ordinal":"int","createdAt":"datetime","updatedAt":"datetime"}},
"rent-payments":{"fields":{"_id":"id","company":"id","project":"id","property":"id","tenant":"id","startDate":"datetime","endDate":"datetime","amount":"number","amountPaid":"number","paymentStatus":"enum:Paid|Due","paymentMode":"enum:Cash|Online|Cheque","generatedReferenceNo":"str","documentSerialNo":"str","paymentReceipt":"str","referenceNo":"str","receivedOn":"datetime","paymentInfo":"array<object>","partialPayments":"array<object>","createdAt":"datetime","updatedAt":"datetime"}},
"tenants":{"fields":{"_id":"id","company":"id","project":"id","property":"id","name":"str","countryCode":"str","phone":"str","email":"str","totalMember":"int","members":"array<object<name:str,age:int,relation:str,_id:id>>","isPets":"bool","bookingDate":"datetime","bookingType":"enum:Temporary Booking|Final Booking","rentAmount":"number","rentIncrement":"number","depositAmount":"number","adjustedDepositAmount":"number","depositPaymentMode":"enum:Cash|Cheque|Online","depositReferenceNo":"str","isMaintenanceIncluded":"bool","isPoliceVerificationDone":"bool","rentStartDate":"datetime","rentAgreementStartDate":"datetime","dueRentAmount":"number","rentPaymentGeneratedOn":"datetime","tenantStatus":"enum:Left|Staying|Temporary Staying","leavingDate":"datetime","createdAt":"datetime","updatedAt":"datetime"}},
"amenities":{"fields":{"_id":"id","name":"str","amenityType":"enum:Land|Project","createdAt":"datetime","updatedAt":"datetime"}}

}
USER-BASED FILTERING:
- Users can only see data from their own company.
- Super admins can see all data.
- Regular users are filtered by company field.
- Use user_id to get user permissions and company access.

FIELD VALUE CLARIFICATIONS:
- In the 'leads' collection, the 'status' field indicates general activity with possible values like 'active', 'inactive'. If a query involves 'active leads', assume 'status' = 'active' unless otherwise specified by the user.
- In the 'leads' collection, the 'leadStatus' field tracks specific stages such as 'New', 'In Progress', 'Closed - Won', 'Closed - Lost'. If a query does not specify a value, consult user or context for clarification.
- For other collections, if a status field’s value (e.g., 'active', 'inactive') is ambiguous in a query, the agent should prompt the user for the exact value or assume common conventions (e.g., 'status' = 'active') while noting this assumption.

AGGREGATION EXAMPLES:
- "Top performing team": Group leads by assignedTo -> users.team, count "Closed - Won" status.
- "Total rent by tenant": Group rent-payments by tenant, sum amounts.
- "Lead conversion rate": Count leads by status, calculate percentages.
- "Revenue by property": Group rent-payments by property, sum amounts.
- "Broker performance": Group broker-payments by broker, sum amounts.
- "Project revenue": Group property-payments by project, sum amounts.
- "Active leads count": Count leads where status = 'active' or leadStatus = 'New' or 'In Progress'.

QUERY EXAMPLES:
- "Show all leads for company X": Query leads where company = company_id.
- "Find users who are super admins": Query companies where superAdmin exists, then get user details.
- "Get total rent paid by tenant": Use aggregation to group and sum.
- "Show properties for a specific project": Query properties where project = project_id.
- "Find all bookings for a lead": Query property-bookings where lead = lead_id.
- "Get all payments for a property": Query property-payments where property = property_id.
- "Count active leads": Query leads where status = 'active' or leadStatus in ['New', 'In Progress'].

NOTES:
- Use only the fields and relationships listed above for queries and aggregations.
- Use $lookup pipelines strictly for documented foreign keys.
- Filter data using documented company, user, and permission constraints.
- If a field value (e.g., for 'status' or 'leadStatus') is not explicitly defined in a query and not clear from context, make a reasonable assumption (e.g., 'status' = 'active' for active leads) and state this assumption in the thought process, or prompt the user for clarification.
- When in doubt, refer to this section for guidance on schema structure and field interpretations.
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
                    user = await db.users.find_one({"_id": ObjectId(user_id)})
                    if not user:
                        return f"Error: User with ID {user_id} not found"
                    
                    user_permissions = user.get("permissions", {})
                    is_super_admin = "super_admin" in user_permissions or "admin" in user_permissions

                    # Only allow access to user's own data unless super admin
                    if not is_super_admin:
                        if collection_name == "users":
                            # Only allow access to own user document
                            if "_id" in mongo_query:
                                if mongo_query["_id"] != ObjectId(user_id):
                                    return "Error: You can only access your own user data"
                            else:
                                mongo_query["_id"] = ObjectId(user_id)
                        elif collection_name in ["leads", "lead-assignments"]:
                            # Only allow access to leads assigned to the user
                            if "assignedTo" in mongo_query:
                                if mongo_query["assignedTo"] != ObjectId(user_id):
                                    return "Error: You can only access leads assigned to you"
                            else:
                                mongo_query["assignedTo"] = ObjectId(user_id)
                        # For all other collections, do NOT filter by company

                    # Permission check (dictionary-based)
                    collection_perms = user_permissions.get(collection_name) or user_permissions.get(f"permissions.{collection_name}")
                    has_permission = False
                    if collection_perms:
                        if "read" in collection_perms or "write" in collection_perms:
                            has_permission = True
                    if "admin" in user_permissions or "super_admin" in user_permissions:
                        has_permission = True

                    if not has_permission:
                        return f"Error: You don't have permission to access {collection_name}. Your permissions: {user_permissions}"

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
               #         has_permission = (
                #            collection_permission in user_permissions or
                 #           read_permission in user_permissions or
                  #          write_permission in user_permissions or
                   #         "admin" in user_permissions or
                    #        "super_admin" in user_permissions
                     #   )
                        
                      #  if not has_permission:
                       #     return f"Error: You don't have permission to access {collection_name}. Required permissions: {collection_permission}, {read_permission}, or {write_permission}. Your permissions: {user_permissions}"
                        user_permissions = user.get("permissions", {})
                        is_super_admin = "super_admin" in user_permissions or "admin" in user_permissions

                        # Check for dictionary-based permissions
                        collection_perms = user_permissions.get(collection_name) or user_permissions.get(f"permissions.{collection_name}")
                        has_permission = False
                        if collection_perms:
                            if "read" in collection_perms or "write" in collection_perms:
                                has_permission = True
                        if "admin" in user_permissions or "super_admin" in user_permissions:
                            has_permission = True

                        if not has_permission:
                            return f"Error: You don't have permission to access {collection_name}. Your permissions: {user_permissions}"

                
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
    description: str = "Get user permissions for filtering queries"
    args_schema: Type[BaseModel] = MongoDBGetUserPermissionsInput

    def _run(self, user_id: str) -> str:
        return "This tool only supports async operations"

    async def _arun(self, user_id: str) -> str:
        try:
            global current_user_context
            if not user_id:
                user_id = current_user_context.get("user_id")
                if not user_id:
                    return "Error: No user_id provided"
            
            # Get user details
            user = await db.users.find_one({"_id": ObjectId(user_id)})
            if not user:
                return f"Error: User with ID {user_id} not found"
            
            permissions = user.get("permissions", [])
            is_super_admin = "super_admin" in permissions or "admin" in permissions
            
            result = {
                "user_id": user_id,
                "user_name": user.get("name"),
                "user_email": user.get("email"),
                "is_super_admin": is_super_admin,
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
        current_user_context["user_id"] = inputs.get("user_id")
        return await super().ainvoke(inputs, config)

agent_executor = CustomAgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)
