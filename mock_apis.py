"""
Mock healthcare APIs for testing/demo purposes.

In production, these would be real API calls to healthcare systems.
For now, just using in-memory data structures.
"""
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from schemas import (
    Patient, InsuranceEligibility, TimeSlot, Appointment,
    InsuranceStatus, AppointmentStatus, PatientGender
)
import random
import string


class MockHealthcareAPIs:
    """Mock healthcare APIs for sandbox testing"""
    
    def __init__(self):
        # Mock patient database
        self.patients: Dict[str, Patient] = {
            "P001": Patient(
                patient_id="P001",
                name="Ravi Kumar",
                date_of_birth="1985-03-15",
                gender=PatientGender.MALE,
                phone="+1-555-0101",
                address="123 Main St, City, State 12345",
                insurance_provider="BlueCross BlueShield"
            ),
            "P002": Patient(
                patient_id="P002",
                name="Priya Sharma",
                date_of_birth="1990-07-22",
                gender=PatientGender.FEMALE,
                phone="+1-555-0102",
                address="456 Oak Ave, City, State 12345",
                insurance_provider="Aetna"
            ),
            "P003": Patient(
                patient_id="P003",
                name="Amit Patel",
                date_of_birth="1978-11-08",
                gender=PatientGender.MALE,
                phone="+1-555-0103",
                address="789 Pine Rd, City, State 12345",
                insurance_provider="UnitedHealthcare"
            ),
            "P004": Patient(
                patient_id="P004",
                name="Sarah Johnson",
                date_of_birth="1992-05-20",
                gender=PatientGender.FEMALE,
                phone="+1-555-0104",
                address="321 Elm St, City, State 12345",
                insurance_provider="Cigna"
            ),
            "P005": Patient(
                patient_id="P005",
                name="Michael Chen",
                date_of_birth="1988-09-12",
                gender=PatientGender.MALE,
                phone="+1-555-0105",
                address="654 Maple Ave, City, State 12345",
                insurance_provider="BlueCross BlueShield"
            ),
            "P006": Patient(
                patient_id="P006",
                name="Emily Rodriguez",
                date_of_birth="1995-12-03",
                gender=PatientGender.FEMALE,
                phone="+1-555-0106",
                address="987 Cedar Blvd, City, State 12345",
                insurance_provider="Aetna"
            ),
            "P007": Patient(
                patient_id="P007",
                name="David Kim",
                date_of_birth="1983-04-25",
                gender=PatientGender.MALE,
                phone="+1-555-0107",
                address="147 Birch Ln, City, State 12345",
                insurance_provider="UnitedHealthcare"
            ),
        }
        
        # Mock insurance eligibility
        self.insurance_data: Dict[str, InsuranceEligibility] = {
            "P001": InsuranceEligibility(
                patient_id="P001",
                insurance_provider="BlueCross BlueShield",
                status=InsuranceStatus.ACTIVE,
                policy_number="BCBS-12345",
                coverage_start="2023-01-01",
                coverage_end="2024-12-31",
                copay_amount=25.0,
                is_eligible=True
            ),
            "P002": InsuranceEligibility(
                patient_id="P002",
                insurance_provider="Aetna",
                status=InsuranceStatus.ACTIVE,
                policy_number="AET-67890",
                coverage_start="2023-06-01",
                coverage_end="2024-12-31",
                copay_amount=30.0,
                is_eligible=True
            ),
            "P003": InsuranceEligibility(
                patient_id="P003",
                insurance_provider="UnitedHealthcare",
                status=InsuranceStatus.EXPIRED,
                policy_number="UHC-11111",
                coverage_start="2022-01-01",
                coverage_end="2023-12-31",
                copay_amount=20.0,
                is_eligible=False
            ),
        }
        
        # Mock booked appointments
        self.appointments: Dict[str, Appointment] = {}
        
    def search_patient(self, name: Optional[str] = None, 
                      patient_id: Optional[str] = None,
                      date_of_birth: Optional[str] = None,
                      phone: Optional[str] = None) -> List[Patient]:
        """Search patients - supports multiple search criteria"""
        results = []
        
        # Exact match by ID first
        if patient_id:
            if patient_id in self.patients:
                results.append(self.patients[patient_id])
            return results
        
        # Otherwise search by other fields
        for patient in self.patients.values():
            match = True
            
            if name and name.lower() not in patient.name.lower():
                match = False
            if date_of_birth and date_of_birth != patient.date_of_birth:
                match = False
            if phone and phone != patient.phone:
                match = False
                
            if match:
                results.append(patient)
        
        return results
    
    def check_insurance_eligibility(self, patient_id: str,
                                   insurance_provider: Optional[str] = None,
                                   service_type: Optional[str] = None) -> InsuranceEligibility:
        """Check if patient's insurance is active and valid"""
        if patient_id not in self.insurance_data:
            # No insurance found
            return InsuranceEligibility(
                patient_id=patient_id,
                insurance_provider=insurance_provider or "Unknown",
                status=InsuranceStatus.INACTIVE,
                is_eligible=False
            )
        
        eligibility = self.insurance_data[patient_id]
        
        # Check specific provider if requested
        if insurance_provider and eligibility.insurance_provider != insurance_provider:
            return InsuranceEligibility(
                patient_id=patient_id,
                insurance_provider=insurance_provider,
                status=InsuranceStatus.INACTIVE,
                is_eligible=False
            )
        
        return eligibility
    
    def find_available_slots(self, department: str,
                            start_date: str,
                            end_date: str,
                            duration_minutes: int = 30) -> List[TimeSlot]:
        """Generate available appointment slots for a date range"""
        slots = []
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        current_date = start
        slot_id_counter = 1
        
        # Provider names by dept (just for demo)
        providers = {
            "cardiology": ["Dr. Smith", "Dr. Johnson"],
            "general": ["Dr. Williams", "Dr. Brown", "Dr. Davis"],
            "orthopedics": ["Dr. Wilson", "Dr. Moore"],
        }
        department_providers = providers.get(department.lower(), ["Dr. Unknown"])
        
        while current_date <= end:
            # 9am-5pm slots, every 30 min
            for hour in range(9, 17):
                for minute in [0, 30]:
                    slot_time = current_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    
                    if slot_time < datetime.now():
                        continue  # Skip past slots
                    
                    # Randomly mark some as unavailable (simulates real booking system)
                    is_available = random.random() > 0.3
                    
                    slot_id = f"SLOT-{department.upper()}-{slot_id_counter:04d}"
                    slot_id_counter += 1
                    
                    end_time = slot_time + timedelta(minutes=duration_minutes)
                    
                    slot = TimeSlot(
                        slot_id=slot_id,
                        department=department,
                        start_time=slot_time.isoformat(),
                        end_time=end_time.isoformat(),
                        provider_name=random.choice(department_providers),
                        available=is_available
                    )
                    
                    slots.append(slot)
            
            current_date += timedelta(days=1)
        
        # Filter to only available slots
        return [s for s in slots if s.available]
    
    def book_appointment(self, patient_id: str,
                        slot_id: str,
                        department: str,
                        reason: Optional[str] = None,
                        appointment_type: str = "follow-up") -> Appointment:
        """Book an appointment - creates appointment record"""
        
        if patient_id not in self.patients:
            raise ValueError(f"Patient {patient_id} not found")
        
        # Generate appointment ID
        appointment_id = f"APT-{''.join(random.choices(string.ascii_uppercase + string.digits, k=8))}"
        
        # For demo, just set time to next week (real system would use slot_id to get actual time)
        start_time = datetime.now() + timedelta(days=7)
        end_time = start_time + timedelta(minutes=30)
        
        appointment = Appointment(
            appointment_id=appointment_id,
            patient_id=patient_id,
            department=department,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            status=AppointmentStatus.BOOKED,
            reason=reason,
            appointment_type=appointment_type,
            provider_name="Dr. Smith",  # TODO: get from slot_id
            created_at=datetime.now().isoformat()
        )
        
        self.appointments[appointment_id] = appointment
        return appointment
