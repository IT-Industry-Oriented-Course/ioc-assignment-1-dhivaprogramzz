"""
Function-calling LLM Agent for Clinical Workflow Automation.

Tried using LangChain's agent executor initially but had issues with HuggingFace models,
so switched to a simpler parsing-based approach. Works well enough for the POC.
"""
import json
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from langchain_core.tools import StructuredTool
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEndpoint
from pydantic import ValidationError

from schemas import (
    SearchPatientInput, CheckInsuranceEligibilityInput,
    FindAvailableSlotsInput, BookAppointmentInput,
    Patient, InsuranceEligibility, TimeSlot, Appointment
)
from mock_apis import MockHealthcareAPIs


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AuditLogger:
    """Simple audit logger - writes to file for compliance"""
    
    def __init__(self, log_file: str = "audit.log"):
        self.log_file = log_file
    
    def log_action(self, action: str, inputs: Dict[str, Any], 
                   outputs: Any, dry_run: bool = False, error: Optional[str] = None):
        """Log everything for audit trail"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "inputs": inputs,
            "outputs": str(outputs) if outputs else None,
            "dry_run": dry_run,
            "error": error
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        logger.info(f"Audit log: {action} - Dry run: {dry_run}")


class ClinicalWorkflowAgent:
    """
    Main agent class - handles natural language requests and calls appropriate functions.
    """
    
    def __init__(self, api_client: MockHealthcareAPIs, 
                 hf_api_key: Optional[str] = None,
                 model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
                 dry_run: bool = False):
        """
        Initialize agent.
        
        Args:
            api_client: API client instance
            hf_api_key: Optional HuggingFace key (works without it)
            model_name: Model to use if key provided
            dry_run: Test mode flag
        """
        self.api_client = api_client
        self.dry_run = dry_run
        self.audit_logger = AuditLogger()
        self.hf_api_key = hf_api_key
        self.model_name = model_name
        
        # Try to init LLM if key provided, but not required
        self.llm = None
        if hf_api_key:
            try:
                self.llm = HuggingFaceEndpoint(
                    endpoint_url=f"https://api-inference.huggingface.co/models/{model_name}",
                    huggingfacehub_api_token=hf_api_key,
                    task="text-generation",
                    temperature=0.1,  # Low temp for more deterministic behavior
                    max_new_tokens=512,
                )
                logger.info("HuggingFace LLM initialized")
            except Exception as e:
                logger.warning(f"Couldn't init HuggingFace LLM: {e}")
                # Falls back to parsing-based approach
        
        # Set up tools
        self.tools = self._create_tools()
        self.tool_map = {tool.name: tool for tool in self.tools}  # Quick lookup
    
    def _create_tools(self) -> List[StructuredTool]:
        """Wrap API functions as LangChain tools"""
        
        def search_patient_wrapper(name: Optional[str] = None,
                                  patient_id: Optional[str] = None,
                                  date_of_birth: Optional[str] = None,
                                  phone: Optional[str] = None) -> str:
            """Search for patients by name, ID, DOB, or phone."""
            try:
                input_data = SearchPatientInput(
                    name=name, patient_id=patient_id,
                    date_of_birth=date_of_birth, phone=phone
                )
                
                if self.dry_run:
                    result = "DRY RUN: Would search for patient"
                    self.audit_logger.log_action(
                        "search_patient",
                        input_data.dict(exclude_none=True),
                        result,
                        dry_run=True
                    )
                    return result
                
                patients = self.api_client.search_patient(
                    name=input_data.name,
                    patient_id=input_data.patient_id,
                    date_of_birth=input_data.date_of_birth,
                    phone=input_data.phone
                )
                
                result = json.dumps([p.dict() for p in patients], indent=2)
                self.audit_logger.log_action(
                    "search_patient",
                    input_data.dict(exclude_none=True),
                    result,
                    dry_run=False
                )
                return result
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                self.audit_logger.log_action(
                    "search_patient",
                    {"name": name, "patient_id": patient_id},
                    None,
                    error=error_msg
                )
                return error_msg
        
        def check_insurance_wrapper(patient_id: str,
                                   insurance_provider: Optional[str] = None,
                                   service_type: Optional[str] = None) -> str:
            """Check insurance eligibility for a patient."""
            try:
                input_data = CheckInsuranceEligibilityInput(
                    patient_id=patient_id,
                    insurance_provider=insurance_provider,
                    service_type=service_type
                )
                
                if self.dry_run:
                    result = f"DRY RUN: Would check insurance for patient {patient_id}"
                    self.audit_logger.log_action(
                        "check_insurance_eligibility",
                        input_data.dict(exclude_none=True),
                        result,
                        dry_run=True
                    )
                    return result
                
                eligibility = self.api_client.check_insurance_eligibility(
                    patient_id=input_data.patient_id,
                    insurance_provider=input_data.insurance_provider,
                    service_type=input_data.service_type
                )
                
                result = json.dumps(eligibility.dict(), indent=2)
                self.audit_logger.log_action(
                    "check_insurance_eligibility",
                    input_data.dict(exclude_none=True),
                    result,
                    dry_run=False
                )
                return result
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                self.audit_logger.log_action(
                    "check_insurance_eligibility",
                    {"patient_id": patient_id},
                    None,
                    error=error_msg
                )
                return error_msg
        
        def find_slots_wrapper(department: str,
                              start_date: str,
                              end_date: str,
                              duration_minutes: int = 30) -> str:
            """Find available appointment slots for a department."""
            try:
                input_data = FindAvailableSlotsInput(
                    department=department,
                    start_date=start_date,
                    end_date=end_date,
                    duration_minutes=duration_minutes
                )
                
                if self.dry_run:
                    result = f"DRY RUN: Would find slots for {department}"
                    self.audit_logger.log_action(
                        "find_available_slots",
                        input_data.dict(),
                        result,
                        dry_run=True
                    )
                    return result
                
                slots = self.api_client.find_available_slots(
                    department=input_data.department,
                    start_date=input_data.start_date,
                    end_date=input_data.end_date,
                    duration_minutes=input_data.duration_minutes
                )
                
                result = json.dumps([s.dict() for s in slots], indent=2)
                self.audit_logger.log_action(
                    "find_available_slots",
                    input_data.dict(),
                    result,
                    dry_run=False
                )
                return result
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                self.audit_logger.log_action(
                    "find_available_slots",
                    {"department": department},
                    None,
                    error=error_msg
                )
                return error_msg
        
        def book_appointment_wrapper(patient_id: str,
                                    slot_id: str,
                                    department: str,
                                    reason: Optional[str] = None,
                                    appointment_type: str = "follow-up") -> str:
            """Book an appointment for a patient."""
            try:
                input_data = BookAppointmentInput(
                    patient_id=patient_id,
                    slot_id=slot_id,
                    department=department,
                    reason=reason,
                    appointment_type=appointment_type
                )
                
                if self.dry_run:
                    result = f"DRY RUN: Would book appointment for patient {patient_id}"
                    self.audit_logger.log_action(
                        "book_appointment",
                        input_data.dict(exclude_none=True),
                        result,
                        dry_run=True
                    )
                    return result
                
                appointment = self.api_client.book_appointment(
                    patient_id=input_data.patient_id,
                    slot_id=input_data.slot_id,
                    department=input_data.department,
                    reason=input_data.reason,
                    appointment_type=input_data.appointment_type
                )
                
                result = json.dumps(appointment.dict(), indent=2)
                self.audit_logger.log_action(
                    "book_appointment",
                    input_data.dict(exclude_none=True),
                    result,
                    dry_run=False
                )
                return result
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                self.audit_logger.log_action(
                    "book_appointment",
                    {"patient_id": patient_id, "slot_id": slot_id},
                    None,
                    error=error_msg
                )
                return error_msg
        
        tools = [
            StructuredTool.from_function(
                func=search_patient_wrapper,
                name="search_patient",
                description="Search for patients by name, patient ID, date of birth, or phone number. Returns a list of matching patients."
            ),
            StructuredTool.from_function(
                func=check_insurance_wrapper,
                name="check_insurance_eligibility",
                description="Check insurance eligibility for a patient. Requires patient_id. Returns eligibility status, coverage dates, and copay information."
            ),
            StructuredTool.from_function(
                func=find_slots_wrapper,
                name="find_available_slots",
                description="Find available appointment slots for a department within a date range. Requires department, start_date (YYYY-MM-DD), and end_date (YYYY-MM-DD)."
            ),
            StructuredTool.from_function(
                func=book_appointment_wrapper,
                name="book_appointment",
                description="Book an appointment for a patient. Requires patient_id, slot_id, and department. Returns the booked appointment details."
            ),
        ]
        
        return tools
    
    def _parse_request(self, user_input: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Parse user input and figure out what functions to call.
        Returns list of (function_name, params) tuples.
        """
        user_lower = user_input.lower()
        actions = []
        
        # Try to extract patient info
        patient_id = None
        patient_name = None
        
        # Look for patient ID like P001, P002, etc
        id_match = re.search(r'\bP\d{3}\b', user_input, re.IGNORECASE)
        if id_match:
            patient_id = id_match.group().upper()
        
        # Extract patient name from request (more flexible)
        # Look for patterns like "patient [Name]", "for [Name]", etc.
        name_patterns = [
            r'patient\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',  # "patient John Smith"
            r'for\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',      # "for Sarah Johnson"
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)',                  # "Ravi Kumar" (standalone)
            r'patient\s+([a-z]+\s+[a-z]+)',                 # "patient sarah johnson" (lowercase)
            r'for\s+([a-z]+\s+[a-z]+)',                     # "for sarah johnson" (lowercase)
            r'\b([a-z]+\s+[a-z]+)\b',                       # "sarah johnson" (standalone lowercase)
        ]
        
        extracted_name = None
        for pattern in name_patterns:
            match = re.search(pattern, user_input)
            if match:
                name = match.group(1)
                # Capitalize first letter of each word
                extracted_name = ' '.join(word.capitalize() for word in name.split())
                break
        
        # If we found a name, use it for search
        if extracted_name:
            patient_name = extracted_name
            # Don't set patient_id here - let search_patient find it
        
        # Extract department
        department = None
        if "cardiology" in user_lower:
            department = "cardiology"
        elif "general" in user_lower:
            department = "general"
        elif "orthopedic" in user_lower:
            department = "orthopedics"
        
        # Parse time references
        start_date = None
        end_date = None
        if "next week" in user_lower:
            start_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
            end_date = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")
        elif "this week" in user_lower:
            start_date = datetime.now().strftime("%Y-%m-%d")
            end_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        elif "tomorrow" in user_lower:
            start_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            end_date = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
        else:
            # Default: next week
            start_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            end_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        
        # Figure out what the user wants to do
        # Check for simple queries like "sarah johnson status" or just a name
        if not any(keyword in user_lower for keyword in ["search", "find", "book", "schedule", "check", "insurance", "slot", "appointment"]):
            # Just a name mentioned - assume they want to search
            if extracted_name or patient_name:
                search_name = patient_name or extracted_name
                actions.append(("search_patient", {"name": search_name}))
        
        if "search" in user_lower and "patient" in user_lower:
            params = {}
            # Use extracted name or patient_name
            search_name = patient_name or extracted_name
            if search_name:
                params["name"] = search_name
            if patient_id:
                params["patient_id"] = patient_id
            # If no specific name/ID but user said "search patient", try to use extracted name
            if not params and extracted_name:
                params["name"] = extracted_name
            if params:
                actions.append(("search_patient", params))
        
        if "insurance" in user_lower or "eligibility" in user_lower or "status" in user_lower:
            if patient_id:
                actions.append(("check_insurance_eligibility", {"patient_id": patient_id}))
            else:
                # Need to search first to get patient_id
                search_name = patient_name or extracted_name
                if search_name:
                    actions.append(("search_patient", {"name": search_name}))
                    actions.append(("check_insurance_eligibility", {"patient_id": "PLACEHOLDER"}))
                    # Will extract patient_id from search result
        
        # Find slots - more flexible matching
        if ("find" in user_lower or "show" in user_lower or "get" in user_lower) and \
           ("slot" in user_lower or "appointment" in user_lower or "available" in user_lower):
            # Default to cardiology if no department specified
            dept = department or "cardiology"
            actions.append(("find_available_slots", {
                "department": dept,
                "start_date": start_date,
                "end_date": end_date
            }))
        
        if ("book" in user_lower or "schedule" in user_lower or "make" in user_lower) and "appointment" in user_lower:
            search_name = patient_name or extracted_name
            dept = department or "general"  # Default department
            
            if patient_id:
                # Find slots first, then book one
                actions.append(("find_available_slots", {
                    "department": dept,
                    "start_date": start_date,
                    "end_date": end_date
                }))
                # Using placeholder for now - would get actual slot_id from find result
                actions.append(("book_appointment", {
                    "patient_id": patient_id,
                    "slot_id": "SLOT-PLACEHOLDER",
                    "department": dept,
                    "reason": "Follow-up appointment",
                    "appointment_type": "follow-up"
                }))
            elif search_name:
                # Need to search for patient first
                actions.append(("search_patient", {"name": search_name}))
                actions.append(("find_available_slots", {
                    "department": dept,
                    "start_date": start_date,
                    "end_date": end_date
                }))
                # Will need patient_id from search result
                actions.append(("book_appointment", {
                    "patient_id": "PLACEHOLDER",  # Will be replaced from search result
                    "slot_id": "SLOT-PLACEHOLDER",
                    "department": dept,
                    "reason": "Appointment",
                    "appointment_type": "follow-up"
                }))
            else:
                # Just booking request without patient info - show available slots
                actions.append(("find_available_slots", {
                    "department": dept,
                    "start_date": start_date,
                    "end_date": end_date
                }))
        
        # Handle complex requests like "schedule X and check Y"
        if "schedule" in user_lower or "book" in user_lower:
            search_name = patient_name or extracted_name
            if search_name or patient_id:
                if not patient_id and search_name:
                    actions.insert(0, ("search_patient", {"name": search_name}))
                
                if department:
                    if not any(a[0] == "find_available_slots" for a in actions):
                        actions.append(("find_available_slots", {
                            "department": department,
                            "start_date": start_date,
                            "end_date": end_date
                        }))
                
                if "insurance" in user_lower or "eligibility" in user_lower:
                    if patient_id:
                        actions.append(("check_insurance_eligibility", {"patient_id": patient_id}))
                    elif search_name:
                        # Already added search above, will get patient_id from result
                        pass
        
        return actions
    
    def process_request(self, user_input: str) -> Dict[str, Any]:
        """
        Process a natural language request.
        
        Args:
            user_input: Natural language request from user
            
        Returns:
            Dictionary with response and metadata
        """
        # Safety check: refuse medical advice requests
        medical_keywords = ["diagnose", "diagnosis", "treatment", "prescribe", 
                          "medication", "disease", "symptom", "cure", "what should",
                          "should i", "can you treat"]
        if any(keyword in user_input.lower() for keyword in medical_keywords):
            return {
                "success": False,
                "response": "I cannot provide medical advice, diagnosis, or treatment recommendations. I can only help with administrative tasks like scheduling appointments, checking insurance, and searching patient records.",
                "error": "Medical advice request refused"
            }
        
        try:
            # Parse request to determine actions
            actions = self._parse_request(user_input)
            
            if not actions:
                # Provide helpful suggestions based on keywords detected
                user_lower = user_input.lower()
                suggestions = []
                if "appointment" in user_lower or "slot" in user_lower:
                    suggestions.append("Try: 'Find available appointment slots' or 'Find cardiology slots'")
                if "book" in user_lower or "schedule" in user_lower:
                    suggestions.append("Try: 'Book appointment for patient [Name]' or 'Schedule appointment for P001'")
                if "patient" in user_lower and "search" not in user_lower:
                    suggestions.append("Try: 'Search for patient [Name]' or 'Search patient P001'")
                if "insurance" in user_lower:
                    suggestions.append("Try: 'Check insurance for patient P001'")
                
                error_msg = "I couldn't understand your request."
                if suggestions:
                    error_msg += "\n\nSuggestions:\n" + "\n".join(f"  - {s}" for s in suggestions)
                else:
                    error_msg += " Please try rephrasing or use specific commands like 'search patient Ravi Kumar' or 'check insurance for patient P001'."
                
                return {
                    "success": False,
                    "response": error_msg
                }
            
            # Execute actions sequentially
            results = []
            context = {}  # Store results for chaining actions
            
            for action_name, params in actions:
                if action_name not in self.tool_map:
                    continue
                
                tool = self.tool_map[action_name]
                
                # Update params with context from previous actions
                # Handle placeholder values for chaining actions
                if action_name == "book_appointment":
                    if "slot_id" in params and params["slot_id"] in ["SLOT-PLACEHOLDER", "PLACEHOLDER"]:
                        if "last_slot_id" in context:
                            params["slot_id"] = context["last_slot_id"]
                        else:
                            results.append(f"Skipping booking: No available slot found")
                            continue
                    
                    if "patient_id" in params and params["patient_id"] in ["PLACEHOLDER"]:
                        if "last_patient_id" in context:
                            params["patient_id"] = context["last_patient_id"]
                        else:
                            results.append(f"Skipping booking: Patient ID not found")
                            continue
                
                if action_name == "check_insurance_eligibility":
                    if "patient_id" in params and params["patient_id"] == "PLACEHOLDER":
                        if "last_patient_id" in context:
                            params["patient_id"] = context["last_patient_id"]
                        else:
                            results.append(f"Skipping insurance check: Patient ID not found")
                            continue
                
                # Execute tool
                try:
                    result = tool.func(**params)
                    results.append(f"{action_name}:\n{result}")
                    
                    # Extract info for chaining actions (e.g., get patient_id from search to use in booking)
                    if action_name == "search_patient":
                        try:
                            if result.startswith("[") or result.startswith("{"):
                                patients = json.loads(result)
                                if patients and len(patients) > 0:
                                    context["last_patient_id"] = patients[0].get("patient_id")
                                    context["last_patient_name"] = patients[0].get("name")
                        except:
                            pass  # Not critical if parsing fails
                    elif action_name == "find_available_slots":
                        try:
                            if result.startswith("[") or result.startswith("{"):
                                slots = json.loads(result)
                                if slots:
                                    context["last_slot_id"] = slots[0].get("slot_id")
                        except:
                            pass
                except Exception as e:
                    results.append(f"Error in {action_name}: {str(e)}")
            
            response = "\n\n".join(results)
            
            return {
                "success": True,
                "response": response,
                "actions_executed": [a[0] for a in actions],
                "metadata": {"context": context}
            }
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {
                "success": False,
                "response": f"Error processing request: {str(e)}",
                "error": str(e)
            }
