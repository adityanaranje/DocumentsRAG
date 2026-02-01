import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class LLMFactory:

    @staticmethod
    def get_llm(model_type="small"):
        """
        Returns a Groq LLM instance based on type.
        """
        api_key = os.getenv("GROQ_API_KEY")
        
        # Groq specific models from environment
        if model_type == "small":
            model_name = os.getenv("GROQ_MODEL_SMALL", "llama-3.1-8b-instant")
        elif model_type == "medium":
            model_name = os.getenv("GROQ_MODEL_MEDIUM", "llama-3.1-8b-instant")
        else:
            model_name = os.getenv("GROQ_MODEL_LARGE", "llama-3.1-8b-instant")
            
        if api_key:
            return ChatGroq(
                model=model_name,
                temperature=0,
                groq_api_key=api_key,
                max_retries=3,  # Automatically retry on rate limits or transient errors
                timeout=30      # Prevent hanging on slow responses
            )
        
        # Fallback to mock for testing without key
        class MockLLM:
            def invoke(self, msg):
                return f"[Groq Mock Response for {model_type}]: Model {model_name} processing..."
        
        return MockLLM()

