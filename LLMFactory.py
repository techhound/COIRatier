from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
import os

class LLMConfig:
  """ The configuration for creating different LLM models as part of the factory."""

  def __init__(self, model_name=None, temperature=1.0, api_token=None, llm_type=None):
      self.model_name = model_name
      self.temperature = temperature
      self.api_token = api_token
      self.llm_type = llm_type

class LLM:
  def connect(self, config: LLMConfig):
      self.config = config

class OpenAI(LLM):
  def connect(self, config: LLMConfig):
      super().connect(config)
      token = config.api_token or os.getenv("OPENAI_API_KEY")
      if not token:
          raise ValueError("OPENAI_API_KEY environment variable is not set.")
      
      return ChatOpenAI(model=config.model_name)

class HuggingFace(LLM):
  def connect(self, config: LLMConfig):
      super().connect(config)
      token = config.api_token or os.getenv("HUGGINGFACEHUB_API_TOKEN")
      if not token:
          raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable is not set.")
      
      return HuggingFaceEndpoint(repo_id=config.model_name, temperature=config.temperature, huggingfacehub_api_token=token)

class LLMFactory:
  def create_llm(self, config: LLMConfig):
      if config.llm_type == "OpenAI":
          return OpenAI()          
      elif config.llm_type == "HuggingFace":
          return HuggingFace()
      else:
          raise ValueError(f"Unknown LLM type: {config.llm_type}")

# Example usage
