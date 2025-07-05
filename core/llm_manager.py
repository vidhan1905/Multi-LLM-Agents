import os
from typing import Dict, Any, Optional
from crewai import LLM as CrewAILLM
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import llm_config

class LLMManager:
    """
    LLM-agnostic manager that handles different providers.
    Uses simple configuration from llm_config.py
    """
    
    def __init__(self):
        self.provider = llm_config.get_provider_name()
        self.config = llm_config.get_current_config()
        
    def get_provider_config(self) -> Dict[str, Any]:
        """Get configuration for the current provider."""
        return self.config
    
    def validate_api_key(self) -> bool:
        """Validate that API key is set for the current provider."""
        api_key_env = self.config["api_key_env"]
        return bool(os.getenv(api_key_env))
    
    def get_crewai_llm(self) -> CrewAILLM:
        """Get CrewAI LLM instance for the current provider."""
        if not self.validate_api_key():
            raise ValueError(f"API key not found for provider '{self.provider}'. Set {self.config['api_key_env']}")
        
        if self.provider == "gemini":
            return CrewAILLM(
                model=self.config["model"],
                api_key=os.getenv(self.config["api_key_env"]),
                temperature=self.config.get("temperature", 0.7)
            )
        elif self.provider == "openai":
            return CrewAILLM(
                model=self.config["model"],
                api_key=os.getenv(self.config["api_key_env"]),
                temperature=self.config.get("temperature", 0.7)
            )
        else:
            raise ValueError(f"Unsupported provider for CrewAI: {self.provider}")
    
    def get_langchain_model_name(self) -> str:
        """Get the model name formatted for LangChain (removes provider prefix)."""
        model_name = self.config["model"]
        if "/" in model_name:
            return model_name.split("/")[-1]
        return model_name

    def get_langchain_llm(self):
        """Get LangChain LLM instance for the current provider."""
        if not self.validate_api_key():
            raise ValueError(f"API key not found for provider '{self.provider}'. Set {self.config['api_key_env']}")
        
        if self.provider == "gemini":
            return ChatGoogleGenerativeAI(
                model=self.get_langchain_model_name(),
                google_api_key=os.getenv(self.config["api_key_env"]),
                temperature=self.config.get("temperature", 0)
            )
        elif self.provider == "openai":
            return ChatOpenAI(
                model=self.get_langchain_model_name(),
                openai_api_key=os.getenv(self.config["api_key_env"]),
                temperature=self.config.get("temperature", 0)
            )
        else:
            raise ValueError(f"Unsupported provider for LangChain: {self.provider}")
    
    def get_current_provider(self) -> str:
        """Get current provider name."""
        return self.provider
    
    def get_current_model(self) -> str:
        """Get current model name."""
        return self.config["model"]
    
    def get_current_temperature(self) -> float:
        """Get current temperature setting."""
        return self.config.get("temperature", 0.7)

    def get_langchain_embeddings(self):
        """Get LangChain embeddings instance for the current provider."""
        if not self.validate_api_key():
            raise ValueError(f"API key not found for provider '{self.provider}'. Set {self.config['api_key_env']}")

        if self.provider == "gemini":
            return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv(self.config["api_key_env"]))
        elif self.provider == "openai":
            # For OpenAI, you might use a specific model like "text-embedding-ada-002"
            # We'll let the library choose the default for now.
            return OpenAIEmbeddings(openai_api_key=os.getenv(self.config["api_key_env"]))
        else:
            raise ValueError(f"Unsupported provider for LangChain embeddings: {self.provider}") 