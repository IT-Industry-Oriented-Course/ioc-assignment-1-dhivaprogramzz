# Clinical Workflow Automation Agent

A function-calling LLM agent for clinical workflow automation that acts as an intelligent coordinator for healthcare administrative tasks. This agent helps with patient search, insurance eligibility checks, appointment scheduling, and care coordination.

##  Important Disclaimer

**This agent does NOT provide medical advice, diagnosis, or treatment recommendations.** It is designed solely for administrative workflow automation tasks like:
- Patient record searches
- Insurance eligibility verification
- Appointment scheduling
- Care coordination

## Features

 **Function Calling**: Uses structured function schemas (FHIR-style) for deterministic API interactions  
 **Safety & Validation**: Input validation, safety checks, and medical advice refusal  
 **Audit Logging**: Comprehensive logging for compliance and auditability  
 **Dry-Run Mode**: Test workflows without executing actual API calls  
 **Mock APIs**: Sandbox healthcare APIs for testing and demonstration  
 **LangChain Integration**: Built with LangChain for LLM orchestration  
 **HuggingFace Support**: Compatible with HuggingFace models (optional)

## Project Structure

```
ioc-cursor/
├── schemas.py              # FHIR-style schemas for data structures
├── mock_apis.py            # Mock/sandbox healthcare APIs
├── agent.py                # LLM agent implementation (original)
├── agent_improved.py       # Improved agent with better parsing
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── audit.log               # Audit log (generated at runtime)
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd ioc-cursor
   ```

2. **Create a virtual environment :**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up HuggingFace API key (optional):**
   
   Create a `.env` file in the project root:
   ```
   HUGGINGFACE_API_KEY=your_api_key_here
   ```
   
   Get your API key from: https://huggingface.co/settings/tokens
   
   **Note**: The agent works without an API key using a fallback execution mode, but full LLM capabilities require a HuggingFace API key.

## Usage

### Basic Usage

Run the application with example requests:

```bash
python main.py
```

### Interactive Mode

Run in interactive mode to process custom requests:

```bash
python main.py --interactive
```

### Dry-Run Mode

Test workflows without executing actual API calls:

```bash
python main.py --dry-run
```

### Custom Request

Process a specific request:

```bash
python main.py "Schedule a cardiology follow-up for patient Ravi Kumar next week and check insurance eligibility"
```

## Example Requests

The agent can handle requests like:

1. **Patient Search:**
   ```
   "Search for patient Ravi Kumar"
   "Find patient with ID P001"
   ```

2. **Insurance Eligibility:**
   ```
   "Check insurance eligibility for patient P001"
   "Verify insurance for Ravi Kumar"
   ```

3. **Find Appointment Slots:**
   ```
   "Find available cardiology appointment slots for next week"
   "Show me available slots in general department"
   ```

4. **Book Appointments:**
   ```
   "Schedule a cardiology follow-up for patient Ravi Kumar next week"
   "Book an appointment for patient P001 in cardiology department"
   ```

5. **Complex Workflows:**
   ```
   "Schedule a cardiology follow-up for patient Ravi Kumar next week and check insurance eligibility"
   ```

## Function Schemas

The agent uses FHIR-style schemas for all data structures:

### Available Functions

1. **`search_patient`**
   - Input: name, patient_id, date_of_birth, phone
   - Output: List of Patient objects

2. **`check_insurance_eligibility`**
   - Input: patient_id, insurance_provider (optional), service_type (optional)
   - Output: InsuranceEligibility object

3. **`find_available_slots`**
   - Input: department, start_date, end_date, duration_minutes (optional)
   - Output: List of TimeSlot objects

4. **`book_appointment`**
   - Input: patient_id, slot_id, department, reason (optional), appointment_type (optional)
   - Output: Appointment object

## Mock Data

The sandbox includes sample patients:

- **P001**: Ravi Kumar (Active insurance)
- **P002**: Priya Sharma (Active insurance)
- **P003**: Amit Patel (Expired insurance)

## Audit Logging

All actions are logged to `audit.log` for compliance tracking. Each log entry includes:
- Timestamp
- Action name
- Input parameters
- Output results
- Dry-run status
- Error messages (if any)

## Safety Features

1. **Medical Advice Refusal**: Automatically refuses requests for medical advice, diagnosis, or treatment
2. **Input Validation**: All inputs are validated against Pydantic schemas
3. **Error Handling**: Comprehensive error handling with detailed error messages
4. **Audit Trail**: All actions are logged for compliance

## Technology Stack

- **LangChain**: LLM orchestration and agent framework
- **HuggingFace**: LLM model integration (optional)
- **Pydantic**: Data validation and schema definition
- **Python**: Core programming language

## Development

### Running Tests

The agent includes mock APIs for testing. You can test individual functions:

```python
from mock_apis import MockHealthcareAPIs

api = MockHealthcareAPIs()
patients = api.search_patient(name="Ravi Kumar")
print(patients)
```

### Extending the Agent

To add new functions:

1. Define input/output schemas in `schemas.py`
2. Implement the function in `mock_apis.py`
3. Add the tool wrapper in `agent_improved.py`
4. Update the parsing logic in `_parse_request()`

## Limitations

- **Mock APIs**: Uses sandbox APIs, not real healthcare systems
- **Simple Parsing**: Uses keyword-based parsing (can be enhanced with better LLM integration)
- **Limited Context**: Basic context chaining between function calls
- **No Real LLM**: Falls back to rule-based execution if HuggingFace API key is not provided

## Future Enhancements

- [ ] Enhanced LLM-based request parsing
- [ ] Integration with real healthcare APIs (FHIR)
- [ ] Multi-turn conversation support
- [ ] Better context management between function calls
- [ ] Support for more complex workflows
- [ ] Integration with MCP (Model Context Protocol)

## License

This project is for educational and demonstration purposes.

## Contact

For questions or issues, please refer to the assignment requirements or contact your instructor.

---

**Remember**: This is a POC (Proof of Concept) for demonstration purposes. In production, you would need to:
- Integrate with real healthcare APIs
- Implement proper authentication and authorization
- Add more robust error handling
- Enhance LLM capabilities
- Implement proper security measures
- Comply with healthcare regulations (HIPAA, etc.)
