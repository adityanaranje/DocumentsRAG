import os
import sqlite3
import json
import hashlib
import time
from typing import Optional, Any
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

class LLMCache:
    """
    Simple SQLite-based cache for LLM responses.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMCache, cls).__new__(cls)
            cls._instance._init_db()
        return cls._instance
    
    def _init_db(self):
        self.db_path = "rag/llm_cache.db"
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS responses (
                id TEXT PRIMARY KEY,
                prompt_hash TEXT,
                model TEXT,
                response TEXT,
                timestamp REAL
            )
        """)
        self.conn.commit()

    def get(self, prompt: str, model: str) -> Optional[str]:
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        cursor = self.conn.execute(
            "SELECT response FROM responses WHERE prompt_hash = ? AND model = ?",
            (prompt_hash, model)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def set(self, prompt: str, model: str, response: str):
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        self.conn.execute(
            "INSERT OR REPLACE INTO responses (id, prompt_hash, model, response, timestamp) VALUES (?, ?, ?, ?, ?)",
            (f"{prompt_hash}_{model}", prompt_hash, model, response, time.time())
        )
        self.conn.commit()

class CachedChatGroq:
    """
    Wrapper around ChatGroq to handle caching and retries.
    """
    def __init__(self, llm_instance, model_name):
        self.llm = llm_instance
        self.model_name = model_name
        self.cache = LLMCache()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception)
    )
    def invoke(self, messages: Any) -> Any:
        # Convert messages to string for cache key
        if isinstance(messages, list):
            prompt_str = json.dumps([m.content for m in messages], sort_keys=True)
        else:
            prompt_str = str(messages)

        # Check cache
        cached_resp = self.cache.get(prompt_str, self.model_name)
        if cached_resp:
            # Reconstruct a mock response object that behaves like the real one
            class MockResponse:
                def __init__(self, content):
                    self.content = content
            return MockResponse(cached_resp)

        # Call API
        try:
            response = self.llm.invoke(messages)
            content = getattr(response, 'content', str(response))
            
            # Cache success
            self.cache.set(prompt_str, self.model_name, content)
            return response
        except Exception as e:
            print(f"[LLM Error] Rate limit or network issue: {e}. Retrying...")
            raise e


class LLMFactory:

    @staticmethod
    def get_llm(complexity="low"):
        """
        Returns a routed and cached LLM instance.
        complexity: "low" (default, instant logic) or "high" (versatile logic)
        """
        api_key = os.getenv("GROQ_API_KEY")
        
        # Default to instant (cost effective)
        default_model = "llama-3.1-8b-instant"
        
        if complexity == "high":
            # For now, map 'versatile' also to 'instant' as per user request to start cheap
            # But keep logic ready to swap to 'llama-3.1-70b-versatile'
            model_name = os.getenv("GROQ_MODEL_LARGE", default_model) 
        else:
            model_name = os.getenv("GROQ_MODEL_SMALL", default_model)
            
        if api_key:
            real_llm = ChatGroq(
                model=model_name,
                temperature=0,
                groq_api_key=api_key,
                # We handle retries in the wrapper, so keep internal retries low
                max_retries=1,
                timeout=30
            )
            return CachedChatGroq(real_llm, model_name)
        
        # Fallback to mock for testing without key
        class MockLLM:
            def invoke(self, msg):
                return f"[Groq Mock Response for {complexity}]: Model {model_name} processing..."
        
        return MockLLM()
