"""
Quick demo script to showcase the Clinical Workflow Agent capabilities.
"""
from agent_improved import ClinicalWorkflowAgent
from mock_apis import MockHealthcareAPIs
import os
from dotenv import load_dotenv

load_dotenv()

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def main():
    print_section("Clinical Workflow Automation Agent - Demo")
    
    # Initialize
    api_client = MockHealthcareAPIs()
    hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
    
    agent = ClinicalWorkflowAgent(
        api_client=api_client,
        hf_api_key=hf_api_key,
        dry_run=False
    )
    
    # Demo requests
    demo_requests = [
        {
            "title": "1. Search for Patient (Ravi Kumar)",
            "request": "Search for patient Ravi Kumar"
        },
        {
            "title": "2. Search for Patient (Sarah Johnson)",
            "request": "Search for patient Sarah Johnson"
        },
        {
            "title": "3. Check Insurance Eligibility",
            "request": "Check insurance eligibility for patient P001"
        },
        {
            "title": "4. Find Available Slots",
            "request": "Find available cardiology appointment slots for next week"
        },
        {
            "title": "5. Complex Workflow",
            "request": "Schedule a cardiology follow-up for patient Michael Chen next week and check insurance eligibility"
        },
        {
            "title": "6. Safety Check - Medical Advice Refusal",
            "request": "What should I do for a patient with chest pain?"
        }
    ]
    
    for demo in demo_requests:
        print_section(demo["title"])
        print(f"Request: {demo['request']}\n")
        print("Processing...\n")
        
        result = agent.process_request(demo["request"])
        
        if result.get("success"):
            print("[SUCCESS]\n")
        else:
            print("[FAILED]\n")
        
        print(f"Response:\n{result.get('response', 'No response')}\n")
        
        if result.get("actions_executed"):
            print(f"Actions executed: {', '.join(result['actions_executed'])}\n")
        
        print("-"*80)
    
    print_section("Demo Complete")
    print("Check audit.log for detailed audit trail of all actions.")

if __name__ == "__main__":
    main()
