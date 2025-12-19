"""
Data schemas for the clinical workflow agent.

Based on FHIR structure but simplified for this POC. Used for validation and type safety.
"""
from typing import Optional, List
from pydantic import BaseModel, Field, validator
from datetime import datetime, date
from enum import Enum


class PatientGender(str, Enum):
    """FHIR Patient Gender values"""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


class AppointmentStatus(str, Enum):
    """FHIR Appointment Status values"""
    PROPOSED = "proposed"
    PENDING = "pending"
    BOOKED = "booked"
    ARRIVED = "arrived"
    FULFILLED = "fulfilled"
    CANCELLED = "cancelled"
    NOSHOW = "noshow"


class InsuranceStatus(str, Enum):
    """Insurance eligibility status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    EXPIRED = "expired"


# Input Schemas for Functions
class SearchPatientInput(BaseModel):
    """Input schema for search_patient function"""
    name: Optional[str] = Field(None, description="Patient full name or partial name")
    patient_id: Optional[str] = Field(None, description="Unique patient identifier")
    date_of_birth: Optional[str] = Field(None, description="Date of birth in YYYY-MM-DD format")
    phone: Optional[str] = Field(None, description="Phone number")
    
    @validator('date_of_birth')
    def validate_date(cls, v):
        if v:
            try:
                datetime.strptime(v, '%Y-%m-%d')
            except ValueError:
                raise ValueError('date_of_birth must be in YYYY-MM-DD format')
        return v


class CheckInsuranceEligibilityInput(BaseModel):
    """Input schema for check_insurance_eligibility function"""
    patient_id: str = Field(..., description="Unique patient identifier")
    insurance_provider: Optional[str] = Field(None, description="Insurance provider name")
    service_type: Optional[str] = Field(None, description="Type of service (e.g., 'cardiology', 'general')")


class FindAvailableSlotsInput(BaseModel):
    """Input schema for find_available_slots function"""
    department: str = Field(..., description="Department name (e.g., 'cardiology', 'general')")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    duration_minutes: Optional[int] = Field(30, description="Appointment duration in minutes")
    
    @validator('start_date', 'end_date')
    def validate_date(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError('Dates must be in YYYY-MM-DD format')
        return v


class BookAppointmentInput(BaseModel):
    """Input schema for book_appointment function"""
    patient_id: str = Field(..., description="Unique patient identifier")
    slot_id: str = Field(..., description="Available slot identifier")
    department: str = Field(..., description="Department name")
    reason: Optional[str] = Field(None, description="Reason for appointment")
    appointment_type: Optional[str] = Field("follow-up", description="Type of appointment")


# Output Schemas
class Patient(BaseModel):
    """FHIR-style Patient resource"""
    patient_id: str = Field(..., description="Unique patient identifier")
    name: str = Field(..., description="Full name")
    date_of_birth: str = Field(..., description="Date of birth in YYYY-MM-DD format")
    gender: PatientGender = Field(..., description="Gender")
    phone: Optional[str] = Field(None, description="Phone number")
    address: Optional[str] = Field(None, description="Address")
    insurance_provider: Optional[str] = Field(None, description="Primary insurance provider")


class InsuranceEligibility(BaseModel):
    """Insurance eligibility information"""
    patient_id: str = Field(..., description="Patient identifier")
    insurance_provider: str = Field(..., description="Insurance provider name")
    status: InsuranceStatus = Field(..., description="Eligibility status")
    policy_number: Optional[str] = Field(None, description="Policy number")
    coverage_start: Optional[str] = Field(None, description="Coverage start date")
    coverage_end: Optional[str] = Field(None, description="Coverage end date")
    copay_amount: Optional[float] = Field(None, description="Copay amount")
    is_eligible: bool = Field(..., description="Whether patient is currently eligible")


class TimeSlot(BaseModel):
    """Available appointment time slot"""
    slot_id: str = Field(..., description="Unique slot identifier")
    department: str = Field(..., description="Department name")
    start_time: str = Field(..., description="Start time in ISO format")
    end_time: str = Field(..., description="End time in ISO format")
    provider_name: Optional[str] = Field(None, description="Provider name")
    available: bool = Field(True, description="Whether slot is available")


class Appointment(BaseModel):
    """FHIR-style Appointment resource"""
    appointment_id: str = Field(..., description="Unique appointment identifier")
    patient_id: str = Field(..., description="Patient identifier")
    department: str = Field(..., description="Department name")
    start_time: str = Field(..., description="Start time in ISO format")
    end_time: str = Field(..., description="End time in ISO format")
    status: AppointmentStatus = Field(..., description="Appointment status")
    reason: Optional[str] = Field(None, description="Reason for appointment")
    appointment_type: str = Field(..., description="Type of appointment")
    provider_name: Optional[str] = Field(None, description="Provider name")
    created_at: str = Field(..., description="Creation timestamp")
