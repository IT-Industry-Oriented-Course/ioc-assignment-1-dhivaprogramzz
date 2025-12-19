"""
Main entry point for Clinical Workflow Automation Agent.
Demonstrates the POC for appointment scheduling and care coordination.
"""
import os
import sys
import json
from dotenv import load_dotenv
from agent_improved import ClinicalWorkflowAgent
from mock_apis import MockHealthcareAPIs


def print_separator():
    """Print a visual separator"""
    print("\n" + "="*80 + "\n")


def print_result(result: dict):
    """Pretty print the result"""
    print_separator()
    if result.get("success"):
        print("✓ SUCCESS")
        print(f"\nResponse:\n{result.get('response', '')}")
    else:
        print("✗ FAILED")
        print(f"\nError: {result.get('error', 'Unknown error')}")
        print(f"\nResponse: {result.get('response', '')}")
    print_separator()


def main():
    """Main application entry point"""
    load_dotenv()
    
    print("="*80)
    print("Clinical Workflow Automation Agent - POC")
    print("="*80)
    print("\nThis agent helps with:")
    print("  • Patient search")
    print("  • Insurance eligibility checks")
    print("  • Finding available appointment slots")
    print("  • Booking appointments")
    print("\n⚠️  IMPORTANT: This agent does NOT provide medical advice or diagnosis.")
    print("="*80)
    
    # Get HuggingFace API key (optional)
    hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not hf_api_key:
        print("\n⚠️  No HUGGINGFACE_API_KEY found in environment.")
        print("   The agent will use a fallback execution mode.")
        print("   To use full LLM capabilities, set HUGGINGFACE_API_KEY in .env file")
    
    # Initialize
    api_client = MockHealthcareAPIs()
    
    # Check for dry-run mode
    dry_run = "--dry-run" in sys.argv or "-d" in sys.argv
    
    if dry_run:
        print("\n🔍 DRY-RUN MODE: No actual API calls will be made\n")
    
    agent = ClinicalWorkflowAgent(
        api_client=api_client,
        hf_api_key=hf_api_key,
        dry_run=dry_run
    )
    
    # Example requests
    example_requests = [
        "Schedule a cardiology follow-up for patient Ravi Kumar next week and check insurance eligibility",
        "Search for patient Ravi Kumar",
        "Check insurance eligibility for patient P001",
        "Find available cardiology appointment slots for next week",
        "Book an appointment for patient P001 in cardiology department"
    ]
    
    if len(sys.argv) > 1 and sys.argv[1] not in ["--dry-run", "-d", "--interactive", "-i"]:
        # Custom request from command line
        user_input = " ".join(sys.argv[1:])
        print(f"\n📝 Processing request: {user_input}\n")
        result = agent.process_request(user_input)
        print_result(result)
    elif "--interactive" in sys.argv or "-i" in sys.argv:
        # Interactive mode
        print("\n💬 Interactive Mode - Type your requests (or 'quit' to exit)\n")
        while True:
            try:
                user_input = input("> ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                if not user_input:
                    continue
                
                result = agent.process_request(user_input)
                print_result(result)
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {e}\n")
    else:
        # Run example requests
        print("\n📋 Running Example Requests\n")
        for i, request in enumerate(example_requests, 1):
            print(f"\nExample {i}: {request}")
            result = agent.process_request(request)
            print_result(result)
        
        print("\n" + "="*80)

        print("="*80)


if __name__ == "__main__":
    main()
