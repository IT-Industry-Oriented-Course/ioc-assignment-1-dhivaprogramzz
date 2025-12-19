# Quick Start Guide

## Installation (5 minutes)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Optional: Set up HuggingFace API key**
   - Create a `.env` file
   - Add: `HUGGINGFACE_API_KEY=your_key_here`
   - Get key from: https://huggingface.co/settings/tokens

## Running the Agent

### Option 1: Run Demo (Recommended for first time)
```bash
python demo.py
```

### Option 2: Run Main Application
```bash
python main.py
```

### Option 3: Interactive Mode
```bash
python main.py --interactive
```

### Option 4: Dry-Run Mode (Test without executing)
```bash
python main.py --dry-run
```

## Example Commands

Try these in interactive mode:

```
> Search for patient Ravi Kumar
> Check insurance eligibility for patient P001
> Find available cardiology appointment slots for next week
> Schedule a cardiology follow-up for patient Ravi Kumar next week and check insurance eligibility
```

## What to Expect

- **Patient Search**: Returns patient information in JSON format
- **Insurance Check**: Returns eligibility status and coverage details
- **Find Slots**: Returns available appointment time slots
- **Book Appointment**: Creates and returns appointment confirmation
- **Medical Advice Requests**: Automatically refused with safety message

## Troubleshooting

**Issue**: "No module named 'langchain'"
- **Solution**: Run `pip install -r requirements.txt`

**Issue**: "HuggingFace API key not found"
- **Solution**: This is optional. The agent works without it using fallback mode.

**Issue**: "No patients found"
- **Solution**: Use the mock patient names: "Ravi Kumar", "Priya Sharma", or patient IDs: "P001", "P002", "P003"

## Next Steps

1. Review `README.md` for detailed documentation
2. Check `audit.log` to see all logged actions
3. Explore `schemas.py` to understand data structures
4. Modify `mock_apis.py` to add more test data
