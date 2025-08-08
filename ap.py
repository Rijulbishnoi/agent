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
SCHEMA_KNOWLEDGE = """
Database Schema and Relationships:... COLLECTIONS AND THEIR FIELDS:
pages: ["__v", "_id", "createdAt", "description", "slug", "status", "title", "updatedAt"]
contact-tags: ["__v", "_id", "company", "createdAt", "description", "name", "status", "updatedAt"]
banks: ["__v", "_id", "bankNameId._id", "bankNameId.icon", "bankNameId.name", "branchAddress", "branchName", "company", "contactPersonDetails.countryCode", "contactPersonDetails.designation", "contactPersonDetails.fullName", "contactPersonDetails.phone", "createdAt", "ifscCode", "status", "swiftCode", "updatedAt"]
plans: ["__v", "_id", "description", "features[]", "maxNoOfProjects", "maxNoOfUsers", "name", "paymentDuration", "permissions.attendance[]", "permissions.bookings[]", "permissions.brokers[]", "permissions.campaigns_email[]", "permissions.campaigns_facebook[]", "permissions.campaigns_instagram[]", "permissions.campaigns_linked_in[]", "permissions.campaigns_payments[]", "permissions.campaigns_sms[]", "permissions.campaigns_templates[]", "permissions.campaigns_twitter[]", "permissions.campaigns_whatsapp[]", "permissions.campaigns_youtube[]", "permissions.chat[]", "permissions.chat_groups[]", "permissions.contacts[]", "permissions.contacts_tags[]", "permissions.contractor_contract_payments[]", "permissions.contractor_contracts[]", "permissions.contractors[]", "permissions.dashboard[]", "permissions.document_miscellaneous[]", "permissions.document_priorities[]", "permissions.finance[]", "permissions.general_expenses[]", "permissions.groups[]", "permissions.lands[]", "permissions.leads[]", "permissions.ledger_payments[]", "permissions.payments[]", "permissions.projects[]", "permissions.properties[]", "permissions.rent[]", "permissions.settings[]", "permissions.subscriptions[]", "permissions.tags[]", "permissions.teams[]", "permissions.tenants[]", "permissions.users[]", "price", "priceInUSD", "slug", "status", "visibleToUser"]
lands: ["__v", "_id", "address", "amenities[]", "city", "company", "country", "createdAt", "currentMarketValue", "isAgricultural", "isAnyConstructionExists", "isOnLease", "leaseAgreementDocs[]", "location.coordinates[]", "location.type", "name", "occupancyStatus", "owners[]->_id", "owners[]->countryCode", "owners[]->email", "owners[]->name", "owners[]->phone", "pastOwners[]->_id", "pastOwners[]->countryCode", "pastOwners[]->email", "pastOwners[]->name", "pastOwners[]->phone", "plotSize", "propertyTaxRefNo", "propertyType[]", "purchasePrice", "rentalIncome", "sizeType", "state", "status", "updatedAt", "zipCode"]
counters: ["__v", "_id", "counts.agreementToSaleDoc", "counts.allotmentLetterDoc", "counts.bankDemandDoc", "counts.bankPaymentReceiptDoc", "counts.bookingPaymentLink", "counts.bookingReceiptDoc", "counts.brokerPayment", "counts.contractPayment", "counts.customerDemandDoc", "counts.customerPaymentReceiptDoc", "counts.finalPossessionLetterDoc", "counts.fullAndFinalSettlementDoc", "counts.generalExpense", "counts.lead", "counts.occupancyCertificateDoc", "counts.possessionOfferLetterDoc", "counts.propertyPayment", "counts.registryOrSaleDeedDoc", "counts.rentPayment", "counts.rentPaymentDoc", "counts.smsPayment", "counts.subscriptionPayment", "counts.task", "counts.welcomeLetterDoc", "createdAt", "updatedAt"]
lead-assignments: ["__v", "_id", "assignee", "company", "createdAt", "defaultPrimary", "defaultSecondary", "lead", "status", "team", "updatedAt"]
general-expenses: ["__v", "_id", "amount", "company", "createdAt", "dateOfPayment", "expenseType", "generatedReferenceNo", "payableAmount", "payeeName", "paymentMode", "referenceNo", "remarks", "status", "taxes", "updatedAt"]
contractor-sub-services: ["__v", "_id", "contractorService", "createdAt", "name", "status", "updatedAt"]
property-bookings: ["__v", "_id", "basicPrice", "bookingAmount", "bookingDate", "bookingPaymentStatus", "bookingType", "clpPhases[]", "clpPhases[]->_id", "clpPhases[]->bankAmount", "clpPhases[]->bankAmountReceived", "clpPhases[]->bankPaymentStatus", "clpPhases[]->customerAmount", "clpPhases[]->customerAmountReceived", "clpPhases[]->customerPaymentStatus", "clpPhases[]->name", "clpPhases[]->ordinal", "clpPhases[]->percentage", "clpPhases[]->status", "company", "contributor", "createdAt", "documentReferences[]->_id", "documentReferences[]->referenceNo", "documentReferences[]->title", "documents[]", "extraCharges[]", "extraCharges[]->_id", "extraCharges[]->amount", "extraCharges[]->chargeFor", "financeDeptStatus", "lead", "paymentTerm", "project", "property", "saleablePrice", "status", "taxAmount", "taxPercent", "timelineDocuments[]->_id", "timelineDocuments[]->document", "timelineDocuments[]->documentLink", "timelineDocuments[]->documentSerialNo", "timelineDocuments[]->documentType", "timelineDocuments[]->priority", "timelineDocuments[]->propertyType", "timelineDocuments[]->templateName", "updatedAt"]
contract-payments: ["__v", "_id", "amount", "company", "contract", "contractor", "createdAt", "dueDate", "generatedReferenceNo", "ordinal", "paymentMode", "paymentStatus", "project", "receivedOn", "referenceNo", "remarks", "status", "updatedAt"]
lead-visited-properties: ["__v", "_id", "company", "createdAt", "lead", "property", "remarks", "status", "updatedAt"]
bhk-types: ["__v", "_id", "createdAt", "name", "status", "updatedAt"]
groups: ["__v", "_id", "company", "createdAt", "members[]", "name", "status", "updatedAt"]
leads: ["__v", "_id", "bhk", "bhkType", "broker", "buyingTimeline", "commissionPercent", "company", "countryCode", "createdAt", "email", "financialQualification", "leadNo", "leadStatus", "maxBudget", "minBudget", "name", "phone", "project", "propertyType", "rotationCount", "secondaryCountryCode", "secondaryPhone", "sourceType", "status", "updatedAt", "whatsAppCountryCode", "whatsAppNumber"]
whatsapp-track: ["__v", "_id", "campaign", "company", "countryCode", "createdAt", "message", "phone", "target", "updatedAt", "whatsAppMessageStatus"]
project-categories: ["__v", "_id", "createdAt", "name", "status", "updatedAt"]
contractor-services: ["__v", "_id", "createdAt", "name", "status", "updatedAt"]
chats: ["__v", "_id", "actions[]", "actions[]->_id", "actions[]->messageStatus", "actions[]->time", "actions[]->user", "company", "createdAt", "group", "message", "messageStatus", "messageType", "receiver", "sender", "status", "updatedAt"]
campaign-payments: ["__v", "_id", "amount", "campaignType", "company", "createdAt", "dueDate", "generatedReferenceNo", "paymentInfo[]", "paymentProcessingStatus", "paymentStatus", "transactionType", "txnId", "updatedAt"]
lead-rotations: ["__v", "_id", "assignee", "company", "createdAt", "date", "lead", "team", "updatedAt"]
property-booking-payment-links: ["__v", "_id", "amount", "company", "createdAt", "dueDate", "generatedReferenceNo", "paymentInfo[]", "paymentStatus", "txnId", "updatedAt"]
agenda-jobs: ["_id", "data.campaign", "data.company", "endDate", "lastFinishedAt", "lastModifiedBy", "lastRunAt", "lockedAt", "name", "nextRunAt", "priority", "repeatInterval", "repeatTimezone", "shouldSaveResult", "skipDays", "startDate", "type"]... properties: ["__v", "_id", "bhk", "bhkType", "blockName", "builtUpArea", "builtUpAreaType", "carpetArea", "carpetAreaType", "company", "createdAt", "facing", "flatNo", "floorName", "furnishedStatus", "images[]", "maxBudget", "minBudget", "noOfBalconies", "noOfBathRooms", "noOfBedRooms", "noOfDrawingRooms", "noOfKitchens", "noOfParkingLots", "project", "propertyArea", "propertyAreaType", "propertyStatus", "propertyType", "propertyUnitSubType", "series", "status", "superBuiltUpArea", "superBuiltUpAreaType", "updatedAt", "vastuCompliant", "videos[]"]
property-unit-sub-types: ["__v", "_id", "createdAt", "name", "projectType", "status", "updatedAt"]
settings: ["__v", "_id", "attendance.isEnabledForAllUsers", "attendance.isFaceRecognitionMandatoryForCheckIn", "attendance.isFaceRecognitionMandatoryForCheckOut", "attendance.isSelfieMandatoryForCheckIn", "attendance.isSelfieMandatoryForCheckOut", "attendance.isShiftTimeEnabledForAllDays", "attendance.isShiftTimingFeatureEnabled", "attendance.shiftTimingDayWise[]", "attendance.shiftTimingDayWise[]->_id", "attendance.shiftTimingDayWise[]->day", "attendance.shiftTimingDayWise[]->isEnabled", "attendance.shiftTimingDayWise[]->shiftEndTime", "attendance.shiftTimingDayWise[]->shiftStartTime", "attendance.userIds[]", "company", "createdAt", "credentials.easeBuzz.credentials[]", "credentials.easeBuzz.credentials[]->_id", "credentials.easeBuzz.credentials[]->clientKey", "credentials.easeBuzz.credentials[]->clientSecret", "credentials.easeBuzz.credentials[]->project", "credentials.easeBuzz.creditCardCharges.charge", "credentials.easeBuzz.creditCardCharges.chargeType", "credentials.easeBuzz.debitCardChargesAbove2000.charge", "credentials.easeBuzz.debitCardChargesAbove2000.chargeType", "credentials.easeBuzz.debitCardChargesBelow2000.charge", "credentials.easeBuzz.debitCardChargesBelow2000.chargeType", "credentials.easeBuzz.neftCharges.charge", "credentials.easeBuzz.neftCharges.chargeType", "credentials.easeBuzz.netBankingCharges.charge", "credentials.easeBuzz.netBankingCharges.chargeType", "credentials.easeBuzz.rtgsCharges.charge", "credentials.easeBuzz.rtgsCharges.chargeType", "credentials.easeBuzz.upiCharges.charge", "credentials.easeBuzz.upiCharges.chargeType", "credentials.easeBuzz.upiCreditCardCharges.charge", "credentials.easeBuzz.upiCreditCardCharges.chargeType", "credentials.instagram.accessToken", "credentials.instagram.accountId", "credentials.ses.credentials.fromEmail", "credentials.ses.credentials.password", "credentials.ses.credentials.userName", "credentials.ses.ratePerEmail", "credentials.smartPing.credentials.dltPrincipalEntityId", "credentials.smartPing.credentials.fromNo", "credentials.smartPing.credentials.password", "credentials.smartPing.credentials.userName", "credentials.smartPing.ratePerSms", "credentials.twilioSendGrid.credentials.apiKey", "credentials.twilioSendGrid.credentials.fromEmail", "credentials.twilioSendGrid.ratePerEmail", "credentials.twilioSms.credentials.accountAuthToken", "credentials.twilioSms.credentials.accountSid", "credentials.twilioSms.credentials.fromNo", "credentials.twilioSms.ratePerSms", "credentials.whatsapp.credentials.appId", "credentials.whatsapp.credentials.authToken", "credentials.whatsapp.credentials.businessAccountId", "credentials.whatsapp.credentials.phoneId", "credentials.whatsapp.ratePerMessage", "emailCampaignService", "general.countryCode", "general.currency[]", "general.currency[]->_id", "general.currency[]->isDefault", "general.currency[]->name", "general.currency[]->symbol", "general.hasCallDetectionActivated", "general.hasCallDetectionForAllLead", "general.hasInternationalSupport", "general.imageHeight", "general.imageWidth", "general.isWaterMarksOnImagesEnabled", "general.opacity", "general.tempPropertyBookingPeriod", "general.tempRentBookingPeriod", "general.waterMarkImage", "general.waterMarkPosition", "isLandExportEnabled", "isProjectExportEnabled", "isPropertyExportEnabled", "leads.assignmentSetPriority[]", "leads.isDuelLeadOwnershipEnabled", "leads.isLeadContactInfoMask", "leads.isLeadSourceEditable", "leads.isLeadsExportEnabled", "leads.leadNotes.isNotesMandatoryEnabled", "leads.leadNotes.isNotesMandatoryOnAddLead", "leads.leadNotes.isNotesMandatoryOnMeetingDone", "leads.leadNotes.isNotesMandatoryOnSiteVisitDone", "leads.leadNotes.isNotesMandatoryOnUpdateLead", "security.isCopyPasteEnabled", "security.isScreenshotEnabled", "security.isTwoFactorAuthenticationEnabled", "smsCampaignService", "timeZone", "updatedAt"]
user-sessions: ["_id", "expires", "session"]
lead-notes: ["__v", "_id", "communicationType", "company", "createdAt", "lead", "meetingDateTime", "nextSiteVisitScheduledDate", "remarks", "siteVisitScheduledDate", "siteVisitStatus", "status", "tag", "updatedAt"]
projects: ["__v", "_id", "address", "amenities[]", "bhkTypesResidential[]", "bhksResidential[]", "blocksCommercial[]", "blocksResidential[]->_id", "blocksResidential[]->blockName", "blocksResidential[]->floors[]->_id", "blocksResidential[]->floors[]->bhk", "blocksResidential[]->floors[]->floorName", "blocksResidential[]->floors[]->noOfUnits", "blocksResidential[]->floors[]->projectUnitSubType", "blocksResidential[]->floors[]->series", "blocksResidential[]->noOfFloors", "brochure[]", "category", "city", "clpPhases[]->_id", "clpPhases[]->name", "clpPhases[]->ordinal", "clpPhases[]->percentage", "clpPhases[]->status", "company", "completionDate", "country", "countryCode", "createdAt", "documents[]", "email", "financerBankDetails[]", "govtApprovedDocuments[]", "images[]", "isGovtApproved", "land", "layoutPlanImages[]", "location.coordinates[]", "location.type", "maxBudget", "minBudget", "name", "nearByLocations[]", "nearByLocations[]->_id", "nearByLocations[]->description", "nearByLocations[]->title", "noOfBlocksResidential", "noOfPhaseResidential", "noOfUnitsResidential", "phone", "projectStatus", "projectType[]", "projectUnitSubType[]", "promoterDetails[]", "propertyUnitSubTypesCommercial[]", "propertyUnitSubTypesResidential[]", "qrCode", "slug", "startDate", "state", "status", "totalBookedUnits", "totalBookingCancelled", "totalCommercial.blocks[]", "totalRefundedAmount", "totalRentAmountRequired", "totalRentRevenue", "totalRentedUnits", "totalResidential.blocks[]->_id", "totalResidential.blocks[]->blockName", "totalResidential.blocks[]->floors[]->_id", "totalResidential.blocks[]->floors[]->floorName", "totalResidential.blocks[]->floors[]->noOfUnits", "totalResidential.blocks[]->floors[]->noOfUnitsConsumed", "totalResidential.blocks[]->floors[]->units[]->_id", "totalResidential.blocks[]->floors[]->units[]->bhk", "totalResidential.blocks[]->floors[]->units[]->noOfUnits", "totalResidential.blocks[]->floors[]->units[]->noOfUnitsConsumed", "totalResidential.blocks[]->floors[]->units[]->projectUnitSubType", "totalResidential.blocks[]->floors[]->units[]->series", "totalResidential.blocks[]->noOfUnits", "totalResidential.blocks[]->noOfUnitsConsumed", "totalSaleAmountRequired", "totalSaleRevenue", "totalSoldUnits", "updatedAt", "videos[]", "zipCode"]
chat-cache: ["__v", "_id", "actions[]", "actions[]->_id", "actions[]->messageStatus", "actions[]->time", "actions[]->user", "company", "createdAt", "group", "message", "messageStatus", "messageType", "receiver", "sender", "status", "updatedAt"]
brokers: ["__v", "_id", "address", "bankDetails.accountNo", "bankDetails.bankAccountType", "bankDetails.bankNameId._id", "bankDetails.bankNameId.icon", "bankDetails.bankNameId.name", "bankDetails.ifscCode", "bankDetails.swiftCode", "city", "commissionPercent", "company", "country", "countryCode", "createdAt", "documentReferences[]->_id", "documentReferences[]->referenceNo", "documentReferences[]->title", "name", "phone", "realEstateLicenseDetails.licenseExpirationDate", "realEstateLicenseDetails.licenseIssueDate", "realEstateLicenseDetails.licenseNo", "state", "status", "updatedAt", "yearStartedInRealEstate", "zipCode"]... contracts: ["__v", "_id", "amount", "anyReasonableExpenses", "chargesOnLatePayment", "company", "contractRate", "contractRateType", "contractor", "createdAt", "endDate", "generatedReferenceNo", "isSubContractAllowed", "milestones[]->_id", "milestones[]->amount", "milestones[]->amountPaid", "milestones[]->amountReceived", "milestones[]->endDate", "milestones[]->name", "milestones[]->ordinal", "milestones[]->paymentStatus", "milestones[]->percentage", "milestones[]->startDate", "milestones[]->status", "payForEachInvoiceDue", "project", "scopeOfWork", "serviceTypes[]->_id", "serviceTypes[]->contractService", "serviceTypes[]->contractSubServices[]", "startDate", "status", "terminationPeriod", "updatedAt", "witnesses[]->_id", "witnesses[]->address", "witnesses[]->countryCode", "witnesses[]->document", "witnesses[]->email", "witnesses[]->name", "witnesses[]->phone"]
bank-names: ["__v", "_id", "createdAt", "icon", "name", "status", "updatedAt"]
countries: ["__v", "_id", "capital", "currency", "currencySymbol", "emoji", "iso2", "name", "phoneCode", "states[]->_id", "states[]->cities[]", "states[]->cities[]->_id", "states[]->cities[]->name", "states[]->name", "states[]->stateCode"]
contractors: ["__v", "_id", "address", "avatar", "bankDetails.accountNo", "bankDetails.bankAccountType", "bankDetails.bankName", "bankDetails.ifscCode", "bankDetails.upiId", "city", "company", "contactPersons[]->_id", "contactPersons[]->countryCode", "contactPersons[]->email", "contactPersons[]->name", "contactPersons[]->phone", "contractorRateType", "contractorType", "country", "countryCode", "createdAt", "documents[]->_id", "documents[]->document", "documents[]->documentRefNo", "documents[]->title", "email", "isGoodsSupplier", "jobTypes[]", "name", "phone", "serviceTypes[]->_id", "serviceTypes[]->contractorService", "serviceTypes[]->contractorSubServices[]", "state", "status", "suitableFor[]", "updatedAt", "yearsOfExperience", "zipCode"]
subscription-payments: ["__v", "_id", "amount", "company", "createdAt", "dueDate", "endDate", "generatedReferenceNo", "paymentInfo[]", "paymentMode", "paymentStatus", "plan", "startDate", "status", "updatedAt"]
sms-track: ["__v", "_id", "campaign", "company", "countryCode", "createdAt", "message", "phone", "smsStatus", "target", "updatedAt"]
attendance: ["__v", "_id", "checkInTime", "checkOutTime", "company", "createdAt", "date", "lastAction", "logs[]->_id", "logs[]->checkInTime", "logs[]->checkOutTime", "logs[]->minutesWorked", "minutesLateIn", "minutesLateOut", "minutesWorked", "status", "updatedAt", "user"]
teams: ["__v", "_id", "company", "createdAt", "defaultPrimary", "defaultSecondary", "groups[]", "members[]", "name", "noOfRotations", "rotationEndTime", "rotationMembers[]", "rotationStartTime", "rotationTime", "status", "teamLead", "updatedAt"]
documents-and-priorities: ["__v", "_id", "company", "createdAt", "document", "documentType", "isMandatory", "paymentTerm", "priority", "propertyType", "status", "templateName", "updatedAt"]
broker-payments: ["__v", "_id", "amount", "booking", "broker", "company", "createdAt", "generatedReferenceNo", "lead", "paidOn", "paymentMode", "paymentStatus", "project", "property", "remarks", "status", "updatedAt"]
rent-payments: ["__v", "_id", "amount", "amountPaid", "company", "createdAt", "documentSerialNo", "endDate", "generatedReferenceNo", "partialPayments[]", "paymentInfo[]", "paymentMode", "paymentReceipt", "paymentStatus", "project", "property", "receivedOn", "remarks", "startDate", "status", "tenant", "updatedAt"]
email-track: ["__v", "_id", "campaign", "company", "createdAt", "email", "emailStatus", "message", "subject", "target", "updatedAt"]
otps: ["__v", "_id", "company", "createdAt", "email", "isVerified", "otpType", "token", "updatedAt", "validTill", "verifyType"]
bhk: ["__v", "_id", "createdAt", "name", "status", "updatedAt"]
contractor-job-types: ["__v", "_id", "createdAt", "name", "status", "updatedAt"]
campaign-templates: ["__v", "_id", "company", "content", "createdAt", "name", "slug", "status", "templateType", "updatedAt", "variables[]", "variables[]->_id", "variables[]->example", "variables[]->variableName", "variables[]->variableType", "whatsAppTemplateCategory", "whatsAppTemplateId", "whatsAppTemplateStatus"]
sms-balance-requests: ["__v", "_id", "campaignPayment", "company", "createdAt", "requestStatus", "status", "updatedAt"]
companies: ["__v", "_id", "addOnServices[]", "address", "allowPlanRenewalOnSamePrice", "amountPaid", "city", "clientName", "contactPersonCountryCode", "contactPersonEmail", "contactPersonName", "contactPersonPhone", "country", "createdAt", "emailBalance", "isSuperAdmin", "logo", "maxNoOfProjects", "maxNoOfUsers", "name", "plan", "planEndDate", "planStartDate", "primaryCountryCode", "primaryEmail", "primaryPhone", "secondaryCountryCode", "secondaryPhone", "smsBalance", "state", "status", "subDomain", "superAdmin", "totalGeneralExpenses", "updatedAt", "whatsAppBalance"]
amenities: ["__v", "_id", "amenityType", "createdAt", "icon", "name", "status", "updatedAt"]
chat-groups: ["__v", "_id", "about", "company", "createdAt", "members[]", "members[]->_id", "members[]->isAdmin", "members[]->user", "name", "status", "updatedAt"]
property-payments: ["__v", "_id", "amount", "booking", "clpPhase", "company", "createdAt", "dueFrom", "financeDeptStatus", "generatedReferenceNo", "ordinal", "paymentMode", "paymentStatus", "paymentType", "project", "property", "receivedOn", "referenceNo", "remarks", "status", "transactionType", "updatedAt"]
permissions: ["_id", "actions[]->name", "actions[]->ordinal", "actions[]->slug", "name", "ordinal", "slug"]
users: ["__v", "_id", "accountType", "askForPasswordReset", "authTokenIssuedAt", "avatar", "canUseSalesMobileApp", "company", "countryCode", "createdAt", "description", "designation", "email", "failedLoginAttempts", "firstName", "formattedPhone", "fullName", "groups[]", "isFake", "isReportingManager", "lastName", "loginPin", "noOfLeadsAssigned", "password", "permissions.attendance[]", "permissions.bookings[]", "permissions.brokers[]", "permissions.campaigns_email[]", "permissions.campaigns_facebook[]", "permissions.campaigns_instagram[]", "permissions.campaigns_linked_in[]", "permissions.campaigns_payments[]", "permissions.campaigns_sms[]", "permissions.campaigns_templates[]", "permissions.campaigns_twitter[]", "permissions.campaigns_whatsapp[]", "permissions.campaigns_youtube[]", "permissions.chat[]", "permissions.chat_groups[]", "permissions.contacts[]", "permissions.contacts_tags[]", "permissions.contractor_contract_payments[]", "permissions.contractor_contracts[]", "permissions.contractors[]", "permissions.dashboard[]", "permissions.document_miscellaneous[]", "permissions.document_priorities[]", "permissions.finance[]", "permissions.general_expenses[]", "permissions.groups[]", "permissions.lands[]", "permissions.leads[]", "permissions.ledger_payments[]", "permissions.payments[]", "permissions.projects[]", "permissions.properties[]", "permissions.rent[]", "permissions.settings[]", "permissions.subscriptions[]", "permissions.tags[]", "permissions.teams[]", "permissions.tenants[]", "permissions.users[]", "phone", "preventLoginTill", "reportsTo", "secondaryCountryCode", "secondaryPhone", "status", "updatedAt"]... miscellaneous-documents: ["__v", "_id", "booking", "company", "createdAt", "name", "project", "property", "status", "updatedAt", "url"]
campaigns: ["__v", "_id", "attachments[]", "billingAmount", "campaignStatus", "campaignType", "cities[]", "company", "countries[]", "createdAt", "csvFileKey", "message", "name", "scheduleTime", "service", "states[]", "status", "tags[]", "targetType", "targets[]", "template", "totalDeliveredCount", "totalSentCount", "updatedAt"]
tags: ["__v", "_id", "company", "createdAt", "description", "icon", "isDefault", "leadCount", "name", "status", "updatedAt"]
tenants: ["__v", "_id", "bookingDate", "bookingType", "company", "countryCode", "createdAt", "documentReferences[]->_id", "documentReferences[]->referenceNo", "documentReferences[]->title", "dueRentAmount", "email", "isMaintenanceIncluded", "isPets", "isPoliceVerificationDone", "members[]->_id", "members[]->age", "members[]->name", "members[]->relation", "name", "phone", "project", "property", "rentAgreementEndDate", "rentAgreementStartDate", "rentAgreements[]", "rentIncrement", "rentPaymentGeneratedOn", "rentStartDate", "status", "tenantStatus", "totalMember", "updatedAt"]
onboarding-requests: ["__v", "_id", "addOnServices[]", "address", "city", "clientName", "contactPersonCountryCode", "contactPersonEmail", "contactPersonName", "contactPersonPhone", "country", "createdAt", "logo", "maxNoOfUsers", "name", "onboardingStatus", "plan", "primaryCountryCode", "primaryEmail", "primaryPhone", "secondaryCountryCode", "secondaryPhone", "state", "subDomain", "updatedAt"]
contacts: ["__v", "_id", "campaignUserType", "company", "countryCode", "createdAt", "email", "formattedPhone", "name", "phone", "status", "tags[]", "updatedAt"]
cold-leads: ["__v", "_id", "coldLeadStatus", "company", "countryCode", "createdAt", "phone", "status", "updatedAt"]
designations: ["__v", "_id", "createdAt", "name", "status", "updatedAt"]

COLLECTIONS AND FOREIGN KEY RELATIONSHIPS:
pages: No foreign keys
contact-tags: company -> companies._id
banks: company -> companies._id
plans: No foreign keys
lands: company -> companies._id, country -> countries._id
counters: No foreign keys
lead-assignments: company -> companies._id, lead -> leads._id, team -> teams._id, defaultPrimary -> users._id, defaultSecondary -> users._id, assignee -> users._id
general-expenses: company -> companies._id
contractor-sub-services: contractorService -> contractor-services._id
property-bookings: company -> companies._id, lead -> leads._id, project -> projects._id, property -> properties._id
contract-payments: company -> companies._id, contractor -> contractors._id, project -> projects._id, contract -> contracts._id
lead-visited-properties: company -> companies._id, lead -> leads._id, property -> properties._id
bhk-types: No foreign keys
groups: company -> companies._id
leads: company -> companies._id, project -> projects._id, broker -> brokers._id, bhk -> bhk._id, bhkType -> bhk-types._id
whatsapp-track: company -> companies._id, campaign -> campaigns._id, target -> contacts._id
project-categories: No foreign keys
contractor-services: No foreign keys
chats: company -> companies._id, sender -> users._id, group -> chat-groups._id
campaign-payments: company -> companies._id
lead-rotations: company -> companies._id, lead -> leads._id, team -> teams._id, assignee -> users._id
property-booking-payment-links: company -> companies._id
agenda-jobs: No foreign keys
properties: company -> companies._id, project -> projects._id, propertyUnitSubType -> property-unit-sub-types._id, bhk -> bhk._id, bhkType -> bhk-types._id
property-unit-sub-types: No foreign keys
settings: company -> companies._id
user-sessions: No foreign keys
lead-notes: company -> companies._id, lead -> leads._id, tag -> tags._id
projects: company -> companies._id, land -> lands._id, category -> project-categories._id, country -> countries._id
chat-cache: group -> chat-groups._id, company -> companies._id, sender -> users._id
brokers: company -> companies._id, country -> countries._id
contracts: company -> companies._id, contractor -> contractors._id, project -> projects._id
bank-names: No foreign keys
countries: No foreign keys
contractors: company -> companies._id, country -> countries._id
subscription-payments: company -> companies._id, plan -> plans._id
sms-track: company -> companies._id, campaign -> campaigns._id, target -> contacts._id
attendance: company -> companies._id, user -> users._id
teams: company -> companies._id, teamLead -> users._id, defaultPrimary -> users._id, defaultSecondary -> users._id
documents-and-priorities: company -> companies._id
broker-payments: company -> companies._id, project -> projects._id, property -> properties._id, booking -> property-bookings._id, lead -> leads._id, broker -> brokers._id
rent-payments: company -> companies._id, project -> projects._id, property -> properties._id, tenant -> tenants._id
email-track: company -> companies._id, campaign -> campaigns._id, target -> contacts._id
otps: company -> companies._id
bhk: No foreign keys
contractor-job-types: No foreign keys
campaign-templates: company -> companies._id
sms-balance-requests: company -> companies._id, campaignPayment -> campaign-payments._id
companies: country -> countries._id, plan -> plans._id, superAdmin -> users._id
amenities: No foreign keys
chat-groups: company -> companies._id
property-payments: company -> companies._id, project -> projects._id, property -> properties._id, booking -> property-bookings._id
permissions: No foreign keys
users: company -> companies._id, designation -> designations._id
miscellaneous-documents: company -> companies._id, project -> projects._id, property -> properties._id, booking -> property-bookings._id
campaigns: company -> companies._id
tags: company -> companies._id
tenants: company -> companies._id, project -> projects._id, property -> properties._id
onboarding-requests: No foreign keys
contacts: company -> companies._id
cold-leads: company -> companies._id
designations: No foreign keys

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
- "Active leads count": Count leads where status = 'active' or leadStatus = 'New' or 'In Progress'.... QUERY EXAMPLES:
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
                        # For all other collections, do NOT filter by company... # Permission check (dictionary-based)
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
