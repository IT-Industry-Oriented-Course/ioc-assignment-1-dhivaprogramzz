# Architecture Overview

## System Design

The Clinical Workflow Automation Agent is built with a modular architecture that separates concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    User Input (Natural Language)            │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              ClinicalWorkflowAgent                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Request Parser (_parse_request)                     │   │
│  │  - Extracts intent and parameters                    │   │
│  │  - Determines which functions to call                │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Safety Checks                                       │   │
│  │  - Medical advice refusal                            │   │
│  │  - Input validation                                  │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Function Executor                                   │   │
│  │  - Sequential function calling                       │   │
│  │  - Context chaining                                  │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Audit Logger                                         │   │
│  │  - Logs all actions                                   │   │
│  │  - Compliance tracking                                │   │
│  └──────────────────────────────────────────────────────┘    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Structured Tools (LangChain)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ search_      │  │ check_       │  │ find_        │     │
│  │ patient      │  │ insurance_   │  │ available_   │     │
│  │              │  │ eligibility  │  │ slots        │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐                                          │
│  │ book_        │                                          │
│  │ appointment  │                                          │
│  └──────────────┘                                          │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              MockHealthcareAPIs (Sandbox)                    │
│  - Patient database                                          │
│  - Insurance eligibility data                                │
│  - Appointment scheduling                                    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              FHIR-style Schemas (Pydantic)                   │
│  - Input validation                                          │
│  - Type safety                                               │
│  - Structured outputs                                        │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Schemas (`schemas.py`)
- **Purpose**: Define FHIR-style data structures
- **Technology**: Pydantic
- **Key Classes**:
  - `Patient`: Patient resource
  - `InsuranceEligibility`: Insurance information
  - `TimeSlot`: Available appointment slots
  - `Appointment`: Appointment resource
  - Input schemas for each function

### 2. Mock APIs (`mock_apis.py`)
- **Purpose**: Simulate healthcare system APIs
- **Features**:
  - In-memory patient database
  - Insurance eligibility simulation
  - Appointment slot generation
  - Appointment booking

### 3. Agent (`agent_improved.py`)
- **Purpose**: Core LLM agent with function calling
- **Key Features**:
  - Natural language parsing
  - Function orchestration
  - Safety checks
  - Audit logging
  - Dry-run support
  - Context chaining between functions

### 4. Main Application (`main.py`)
- **Purpose**: Entry point and CLI interface
- **Modes**:
  - Example mode (default)
  - Interactive mode
  - Dry-run mode
  - Custom request mode

## Data Flow

1. **User Input** → Natural language request
2. **Parsing** → Extract intent and parameters
3. **Validation** → Check safety and validate inputs
4. **Function Selection** → Determine which functions to call
5. **Execution** → Call functions sequentially with context
6. **Audit Logging** → Log all actions
7. **Response** → Return structured results

## Safety Mechanisms

1. **Medical Advice Refusal**
   - Keyword detection
   - Automatic refusal with explanation

2. **Input Validation**
   - Pydantic schema validation
   - Type checking
   - Required field validation

3. **Error Handling**
   - Try-catch blocks
   - Graceful degradation
   - Detailed error messages

4. **Audit Trail**
   - All actions logged
   - Timestamps
   - Input/output tracking
   - Error logging

## Extension Points

### Adding New Functions

1. Define input/output schemas in `schemas.py`
2. Implement function in `mock_apis.py`
3. Create tool wrapper in `agent_improved.py`
4. Update parsing logic in `_parse_request()`

### Integrating Real APIs

Replace `MockHealthcareAPIs` with real API client:
- Implement same interface
- Handle authentication
- Add retry logic
- Implement rate limiting

### Enhancing LLM Integration

- Use HuggingFace API for better parsing
- Implement few-shot learning
- Add conversation memory
- Support multi-turn dialogues

## Compliance Considerations

- **Audit Logging**: All actions logged to `audit.log`
- **No Medical Advice**: Hard-coded refusal mechanism
- **Input Validation**: Schema-based validation
- **Error Tracking**: Comprehensive error logging
- **Dry-Run Mode**: Test without side effects

## Performance Considerations

- **Sequential Execution**: Functions called one at a time
- **Context Caching**: Results stored for chaining
- **Lazy LLM Loading**: LLM only loaded if API key provided
- **Efficient Parsing**: Keyword-based parsing (fast, but limited)

## Future Enhancements

1. **Parallel Function Execution**: Execute independent functions in parallel
2. **Better LLM Integration**: Use function calling APIs properly
3. **Conversation Memory**: Remember previous interactions
4. **Multi-step Planning**: Plan complex workflows before execution
5. **Real-time Updates**: WebSocket support for real-time updates
6. **FHIR Integration**: Direct FHIR API integration
