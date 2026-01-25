
import sys
import os
import asyncio

# Add current directory to path so we can import nursesim_rl
sys.path.append(os.getcwd())

from nursesim_rl.pds_client import PDSClient, PDSEnvironment

async def main():
    print("🏥 Testing NHS PDS Client...")
    
    # 1. Test Verification
    print("\n1. Testing NHS Number Validation")
    valid = "9000000009"
    invalid = "1234567890"
    
    if PDSClient.validate_nhs_number(valid):
        print(f"✅ Valid number {valid} passed")
    else:
        print(f"❌ Valid number {valid} FAILED")
        
    if not PDSClient.validate_nhs_number(invalid):
        print(f"✅ Invalid number {invalid} passed (rejected)")
    else:
        print(f"❌ Invalid number {invalid} FAILED (accepted)")

    # 2. Test Sandbox Lookup
    print("\n2. Testing Sandbox API Lookup (Network Request)")
    client = PDSClient(environment=PDSEnvironment.SANDBOX)
    
    try:
        print(f"   Looking up {valid}...")
        patient = await client.lookup_patient(valid)
        print(f"✅ Success! Found patient:")
        print(f"   Name: {patient.full_name}")
        print(f"   Age: {patient.age}")
        print(f"   Gender: {patient.gender}")
        print(f"   GP: {patient.gp_practice_name}")
    except Exception as e:
        print(f"❌ API Lookup Failed: {e}")
        if hasattr(e, 'response'):
             print(f"Response Body: {e.response.text}")

if __name__ == "__main__":
    asyncio.run(main())
