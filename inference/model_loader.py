
import os
import getpass
import openai
from langchain_community.llms import OpenLLM
from langchain_openai import ChatOpenAI
"""Note: uncomment these codes to use these models, lookup this documentaion page: https://python.langchain.com/docs/integrations/chat/
download the packages also from there"""
# from langchain_anthropic import ChatAnthropic
# from langchain_mistralai import ChatMistralAI
# from langchain_google_genai import ChatGoogleGenerativeAI


class BaseModel:
    """
    Loads an LLM based on the specified provider (e.g. OpenAI).
    """

    def __init__(self, llm_provider, model_name, temperature, max_tokens):
        """
        :param llm_provider: e.g. "OpenAI"
        :param model_name:   e.g. "gpt-3.5-turbo" or "gpt-4"
        :param temperature:  float in [0,2]
        :param max_tokens:   how many tokens the LLM can generate in output
        """
        self.llm_provider = llm_provider
        self.model_name   = model_name
        self.temperature  = temperature
        self.max_tokens   = max_tokens
        self.model        = None

    def load(self):
        """Instantiate the LLM based on provider."""
        if self.llm_provider == "OpenAI":
            # Check if OPENAI_API_KEY is set, else prompt for it:
            if "OPENAI_API_KEY" not in os.environ:
                os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter OpenAI API key: ")

            self.model = ChatOpenAI(
                openai_api_key=os.environ["OPENAI_API_KEY"],
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        # elif self.llm_provider == "ANTHROPIC":
        #     if "ANTHROPIC_API_KEY" not in os.environ:
        #         os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter Anthropic API key: ")
                
        #     self.model = ChatAnthropic(
        #         anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
        #         model=self.model_name,
        #         temperature=self.temperature,
        #         max_tokens=self.max_tokens
        #     )
        # elif self.llm_provider == "MISTRAL":
        #     if "MISTRAL_API_KEY" not in os.environ:
        #         os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter Mistral API key: ")
        #     self.model = ChatMistralAI(
        #         mistral_api_key=os.environ["MISTRAL_API_KEY"],
        #         model=self.model_name,
        #         temperature=self.temperature,
        #         max_tokens=self.max_tokens
        #     )
        # elif self.llm_provider == "GOOGLE":
        #     if "GOOGLE_API_KEY" not in os.environ:
        #         os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ") 
        #     self.model = ChatGoogleGenerativeAI(
        #         google_api_key=os.environ["GOOGLE_API_KEY"],
        #         model=self.model_name,
        #         temperature=self.temperature,
        #         max_tokens=self.max_tokens
        #     )
        elif self.llm_provider == "Custom":
            server_url = input("Enter the server URL: ")
            self.model = OpenLLM(server_url=server_url, api_key="na")
        if self.model is None:
            raise Exception(f"LLM '{self.llm}' is not supported. Check LangChain documentation to implement.")
        

    def get_model(self):
        """Return the loaded model (LangChain's ChatOpenAI or similar)."""
        return self.model
