import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env in root
load_dotenv()

CACHE_FILE = os.path.join("data", "llm_cache.json")

# Load existing cache
_cache = {}
if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            _cache = json.load(f)
    except Exception as e:
        print(f"[Cache Error] Warning: Could not load cache: {e}")

def save_cache():
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_cache, f, indent=2)
    except Exception as e:
        print(f"[Cache Error] Warning: Could not save cache: {e}")

# Verify keys and configure client
openrouter_key = os.environ.get("OPENROUTER_API_KEY")
gemini_key = os.environ.get("GEMINI_API_KEY")

if openrouter_key:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_key
    )
    is_openrouter = True
    print("[LLM Client] Using OpenRouter API")
else:
    client = OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=gemini_key
    )
    is_openrouter = False
    print("[LLM Client] Using Direct Gemini API")

_last_api_call_time = 0.0

def get_chat_completion(model: str, messages: list, temperature: float = 0.1, max_tokens: int = 500) -> str:
    """Wrapper around LLM completion with local file caching and backoff on rate limit and transient errors."""
    global _last_api_call_time
    
    original_model = model
    if is_openrouter:
        model = "google/gemini-2.5-flash-lite"
    else:
        model = "gemini-3.1-flash-lite"

    # Create cache key (based on original model configuration)
    cache_key_dict = {
        "model": original_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    # Sort keys for deterministic JSON serialization
    cache_key = json.dumps(cache_key_dict, sort_keys=True)

    if cache_key in _cache:
        cached_val = _cache[cache_key]
        # Make sure it isn't an error response or empty answer cached previously
        is_bad = (
            not isinstance(cached_val, str)
            or not cached_val.strip()
            or "Error generating answer" in cached_val
            or "rate limit" in cached_val.lower()
            or "429" in cached_val
            or "503" in cached_val
        )
        if not is_bad:
            return cached_val

    max_retries = 5
    initial_delay = 5.0
    backoff_factor = 2.0

    for attempt in range(max_retries):
        try:
            now = time.time()
            elapsed = now - _last_api_call_time
            spacing = 1.0 if is_openrouter else 4.0
            if elapsed < spacing:
                sleep_needed = spacing - elapsed
                time.sleep(sleep_needed)

            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            _last_api_call_time = time.time()
            
            content = resp.choices[0].message.content
            text = content.strip() if content is not None else ""
            
            # Save to cache
            _cache[cache_key] = text
            save_cache()
            return text

        except Exception as e:
            err_msg = str(e)
            print(f"[LLM Error] Call attempt {attempt+1} failed: {err_msg}")
            
            is_transient = (
                "429" in err_msg
                or "503" in err_msg
                or "500" in err_msg
                or "rate limit" in err_msg.lower()
                or "tpd" in err_msg.lower()
                or "tpm" in err_msg.lower()
                or "overloaded" in err_msg.lower()
                or "unavailable" in err_msg.lower()
                or "demand" in err_msg.lower()
            )
            
            if is_transient and attempt < max_retries - 1:
                delay = initial_delay * (backoff_factor ** attempt)
                print(f"[LLM Retry] Sleeping for {delay:.1f} seconds before retrying...")
                time.sleep(delay)
            else:
                raise e


