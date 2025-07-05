# LLM Configuration File
# Change these settings to use different LLM providers and models

# =============================================================================
# LLM PROVIDER CONFIGURATION
# =============================================================================

# Available providers: "gemini", "openai"
# Change this to switch between different LLM providers
CURRENT_PROVIDER = "gemini"

# =============================================================================
# PROVIDER-SPECIFIC SETTINGS
# =============================================================================

# Gemini Configuration
GEMINI_CONFIG = {
    "model": "gemini/gemini-2.0-flash",
    "temperature": 0,
    "api_key_env": "GEMINI_API_KEY"
}

# OpenAI Configuration
OPENAI_CONFIG = {
    "model": "openai/gpt-4o-mini",
    "temperature": 0,
    "api_key_env": "OPENAI_API_KEY"
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_current_config():
    """Get the current provider configuration."""
    if CURRENT_PROVIDER == "gemini":
        return GEMINI_CONFIG
    elif CURRENT_PROVIDER == "openai":
        return OPENAI_CONFIG
    else:
        raise ValueError(f"Unsupported provider: {CURRENT_PROVIDER}")

def get_provider_name():
    """Get the current provider name."""
    return CURRENT_PROVIDER

# =============================================================================
# USAGE INSTRUCTIONS
# =============================================================================

"""
HOW TO USE THIS CONFIGURATION FILE:

1. To change LLM provider:
   - Change CURRENT_PROVIDER to "gemini" or "openai"
   - Make sure you have the corresponding API key in your .env file

2. To change model:
   - Modify the "model" field in the respective provider config
   - For Gemini: "gemini/gemini-2.0-flash", "gemini/gemini-1.5-pro"
   - For OpenAI: "openai/gpt-4o-mini", "openai/gpt-4", "openai/gpt-3.5-turbo"

3. To change temperature:
   - Modify the "temperature" field (0.0 = deterministic, 1.0 = creative)

FRAMEWORK USAGE:
- Trip Planning Agent: Uses CrewAI framework
- RAG Agent: Uses LangChain framework  
- SQL Agent: Uses LangChain framework
- Pandas Agent: Uses LangChain framework

Each agent automatically uses the appropriate framework for its task.

EXAMPLE CHANGES:

# To use OpenAI GPT-4:
CURRENT_PROVIDER = "openai"
OPENAI_CONFIG["model"] = "gpt-4"

# To use Gemini with higher creativity:
CURRENT_PROVIDER = "gemini"
GEMINI_CONFIG["temperature"] = 0.7
""" 