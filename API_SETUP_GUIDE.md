# 🔑 API Setup Guide for BLOOMS BOT

## What is an API?

**API** stands for **Application Programming Interface**. Think of it like a **restaurant menu**:

- **You (your code)** = Customer
- **API** = Menu with available dishes
- **API Server (OpenAI)** = Kitchen that prepares the food
- **API Key** = Your ID card that proves you're allowed to order

When your code calls an API, it's like ordering food:
1. You send a **request** (what you want)
2. The server processes it (prepares the food)
3. You get back a **response** (your order)

## What is an API Key?

An **API Key** is like a **password** that proves:
- ✅ You have permission to use the service
- ✅ You have an account (usually paid)
- ✅ The service can track your usage and bill you

**IMPORTANT**: API keys are **SECRET**! Never share them publicly or commit them to GitHub.

## How BLOOMS BOT Uses the OpenAI API

### Step-by-Step Flow:

1. **Your code** (`gpt_question_gen.py`) wants to generate questions
2. It reads your **API key** from the `.env` file
3. It sends your **syllabus text** + **instructions** to OpenAI's servers
4. OpenAI's AI model (GPT) **generates questions** based on your text
5. OpenAI sends back **JSON with questions**
6. Your code receives and processes the questions

### Where the API Key is Used:

Look at `gpt_question_gen.py`, line 42-48:

```python
def _get_client() -> "OpenAI":
    if OpenAI is None:
        raise RuntimeError("openai package is not installed...")
    api_key = os.getenv(API_KEY_ENV)  # ← Reads API key from .env file
    if not api_key:
        raise RuntimeError(f"{API_KEY_ENV} is not set...")  # ← Error if missing!
    return OpenAI(api_key=api_key)  # ← Uses API key to authenticate
```

Then on line 148, the code makes the actual API call:

```python
response = client.chat.completions.create(
    model=GEN_MODEL,  # Which AI model to use
    temperature=temperature,  # How creative (low = more focused)
    messages=[...],  # Your prompt/question
)
```

This is where your code **talks to OpenAI's servers** over the internet!

## How to Get Your OpenAI API Key

### Step 1: Create an OpenAI Account

1. Go to **https://platform.openai.com/**
2. Click **"Sign Up"** or **"Log In"**
3. Create an account (you'll need an email and phone number)

### Step 2: Add Payment Method

⚠️ **IMPORTANT**: OpenAI charges per API call (very small amounts, but you need a payment method)

1. Go to **Settings** → **Billing**
2. Click **"Add Payment Method"**
3. Add a credit/debit card

### Step 3: Generate API Key

1. Go to **https://platform.openai.com/api-keys**
2. Click **"Create new secret key"**
3. Give it a name (e.g., "BLOOMS BOT")
4. **COPY THE KEY IMMEDIATELY** - you won't see it again!
   - It looks like: `sk-proj-...` (starts with `sk-`)

### Step 4: Set Up Your `.env` File

1. In your project folder (`BLOOMSBOT 2`), create a file named `.env`
2. Open it in a text editor
3. Add this line (replace with YOUR actual key):

```env
OPENAI_API_KEY=your_api_key_here
BLOOMSBOT_GEN_MODEL=gpt-4o-mini
```

**Note:** Replace `your_api_key_here` with your actual API key from OpenAI.

### Step 5: Verify It Works

Run this test:

```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('API Key found!' if os.getenv('OPENAI_API_KEY') else 'API Key NOT found!')"
```

If you see "API Key found!", you're good! ✅

## Common Errors and Solutions

### Error: `OPENAI_API_KEY is not set in the environment or .env file`

**Solution:**
- Make sure you created a `.env` file (not `.env.txt` or `env.txt`)
- Make sure it's in the same folder as `app.py`
- Make sure the line starts with `OPENAI_API_KEY=` (no spaces around `=`)

### Error: `Invalid API key`

**Solution:**
- Check that you copied the entire key (they're very long!)
- Make sure there are no extra spaces or quotes
- Try generating a new key from OpenAI's website

### Error: `Insufficient quota` or `Rate limit exceeded`

**Solution:**
- You've hit OpenAI's usage limits
- Check your billing on OpenAI's website
- Wait a few minutes and try again

## Security Best Practices

✅ **DO:**
- Keep your `.env` file **local only** (never upload to GitHub)
- Add `.env` to `.gitignore` if using Git
- Use different API keys for different projects

❌ **DON'T:**
- Share your API key with anyone
- Commit `.env` to version control
- Hardcode API keys in your Python files
- Post screenshots with your API key visible

## Cost Estimation

OpenAI charges based on:
- **Model used** (gpt-4o-mini is cheaper than gpt-4)
- **Tokens** (words/characters sent + received)

**Rough estimate for BLOOMS BOT:**
- Small syllabus (1-2 pages): ~$0.01-0.05 per generation
- Medium syllabus (5-10 pages): ~$0.05-0.20 per generation
- Large syllabus (20+ pages): ~$0.20-0.50 per generation

**Tip**: Start with `gpt-4o-mini` (cheaper) and upgrade if quality isn't good enough.

## Testing Your Setup

After setting up your `.env` file, test the API connection:

```python
# test_api.py
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ API Key not found!")
else:
    print("✅ API Key found!")
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'Hello, BLOOMS BOT!'"}],
            max_tokens=10,
        )
        print(f"✅ API works! Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ API Error: {e}")
```

Save this as `test_api.py` and run:
```bash
python test_api.py
```

If you see "✅ API works!", everything is set up correctly!

---

## Summary

1. **API** = A way for your code to talk to external services (like OpenAI)
2. **API Key** = Your secret password to authenticate with the service
3. **How it works**: Your code → sends request with API key → OpenAI → sends back AI-generated questions
4. **Setup**: Get key from OpenAI → Put in `.env` file → Code reads it automatically
5. **Security**: Never share your API key!

Now you're ready to use BLOOMS BOT! 🚀

