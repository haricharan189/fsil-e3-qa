import os
import getpass
import openai
from langchain_community.llms import OpenLLM
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI


class BaseModel:
    """
    Loads an LLM based on the specified provider (OpenAI, Anthropic, Mistral, Google, 
    or OpenLLM in either local-inprocess or server mode).
    """

    def __init__(self, llm_provider, model_name, temperature, max_tokens):
        """
        :param llm_provider: e.g. "OpenAI", "ANTHROPIC", "MISTRAL", "GOOGLE",
                             "OPENLLM_LOCAL", or "Custom"
        :param model_name:   e.g. "gpt-3.5-turbo", "gpt-4", or "dolly-v2"
        :param temperature:  float in [0,2]
        :param max_tokens:   how many tokens the LLM can generate in output
        """
        self.llm_provider = llm_provider
        self.model_name   = model_name
        self.temperature  = temperature
        self.max_tokens   = max_tokens
        self.model        = None

    def load(self):
        """Instantiate the LLM based on the specified provider."""
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

        elif self.llm_provider == "ANTHROPIC":
            if "ANTHROPIC_API_KEY" not in os.environ:
                os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter Anthropic API key: ")
            self.model = ChatAnthropic(
                anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

        elif self.llm_provider == "MISTRAL":
            if "MISTRAL_API_KEY" not in os.environ:
                os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter Mistral API key: ")
            self.model = ChatMistralAI(
                mistral_api_key=os.environ["MISTRAL_API_KEY"],
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

        elif self.llm_provider == "GOOGLE":
            if "GOOGLE_API_KEY" not in os.environ:
                os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")
            self.model = ChatGoogleGenerativeAI(
                google_api_key=os.environ["GOOGLE_API_KEY"],
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

        elif self.llm_provider == "OPENLLM_LOCAL":
            """
            This uses OpenLLM in an 'embedded' or in-process way, 
            loading the model locally. For example, if your model_name = "dolly-v2", 
            you might want to specify model_id or other optional params. 
            
            By default, we pass the same name for both model_name & model_id. 
            If you want a different model_id (e.g. "databricks/dolly-v2-3b"), 
            consider changing the code or storing it in config.
            """
            # Possibly override model_id in config or environment if needed
            self.model = OpenLLM(
                model_name=self.model_name,  # e.g. "dolly-v2"
                model_id=self.model_name,    # e.g. "databricks/dolly-v2-3b"
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

        elif self.llm_provider == "Custom":
            """
            This uses OpenLLM with a 'server_url', which can be a local or remote 
            endpoint you started with `openllm start ...`. 
            Example: "http://localhost:3000"
            """
            server_url = input("Enter the server URL (e.g. http://localhost:3000): ")
            self.model = OpenLLM(server_url=server_url, api_key="na")

        else:
            raise ValueError(
                f"LLM provider '{self.llm_provider}' not supported. "
                "Choose from: OpenAI, ANTHROPIC, MISTRAL, GOOGLE, OPENLLM_LOCAL, or Custom."
            )

        if self.model is None:
            raise RuntimeError(
                f"Failed to load LLM for provider '{self.llm_provider}'. "
                "Check your configuration or provider logic."
            )

    def get_model(self):
        """Return the loaded LangChain-compatible model (ChatOpenAI, ChatAnthropic, etc.)."""
        return self.model
