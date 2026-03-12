"""
Simple test script to verify your OpenAI API key is set up correctly.

Run this before using BLOOMS BOT to make sure everything works:
    python test_api.py
"""

from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

def test_api_setup():
    """Test if API key is configured and working."""
    print("🔍 Testing OpenAI API Setup...\n")
    
    # Check if API key exists
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: API Key not found!")
        print("\n📝 How to fix:")
        print("1. Create a file named '.env' in this folder")
        print("2. Add this line: OPENAI_API_KEY=your_actual_key_here")
        print("3. Get your key from: https://platform.openai.com/api-keys")
        return False
    
    if api_key == "your_api_key_here" or not api_key.startswith("sk-"):
        print("❌ ERROR: API Key looks invalid!")
        print(f"   Found: {api_key[:10]}...")
        print("\n📝 Make sure you replaced 'your_api_key_here' with your real key")
        return False
    
    print(f"✅ API Key found: {api_key[:10]}...{api_key[-4:]}")
    
    # Test API connection
    try:
        client = OpenAI(api_key=api_key)
        print("🔄 Testing API connection...")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Say 'Hello, BLOOMS BOT!' in exactly 3 words."}
            ],
            max_tokens=10,
        )
        
        result = response.choices[0].message.content
        print(f"✅ API works! Response: {result}")
        print("\n🎉 Everything is set up correctly! You can now use BLOOMS BOT.")
        return True
        
    except Exception as e:
        print(f"❌ API Error: {e}")
        print("\n📝 Common issues:")
        print("- Invalid API key (check OpenAI dashboard)")
        print("- No payment method added (add card on OpenAI)")
        print("- Insufficient credits")
        return False

if __name__ == "__main__":
    test_api_setup()

