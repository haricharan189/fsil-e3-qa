import os
import getpass
import openai

# For open-source LLM usage (OpenLLM)
from langchain_community.llms import OpenLLM

# For proprietary LLM usage
from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
# from langchain_mistralai import ChatMistralAI
# from langchain_google_genai import ChatGoogleGenerativeAI


class BaseModel:
    """
    Loads an LLM based on the specified provider (OpenAI, Anthropic, Mistral, Google,
    or OpenLLM via local or custom server).
    """

    def __init__(self, llm_provider, model_name, temperature, max_tokens):
        """
        :param llm_provider: e.g. "OpenAI", "ANTHROPIC", "MISTRAL", "GOOGLE",
                             "OPENLLM_LOCAL", or "Custom"
        :param model_name:   e.g. "gpt-3.5-turbo", "dolly-v2", "falcon-7b-instruct", etc.
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
            # For official OpenAI
            if "OPENAI_API_KEY" not in os.environ:
                os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter OpenAI API key: ")

            self.model = ChatOpenAI(
                openai_api_key=os.environ["OPENAI_API_KEY"],
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

        # elif self.llm_provider == "ANTHROPIC":
        #     # For Anthropic
        #     if "ANTHROPIC_API_KEY" not in os.environ:
        #         os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter Anthropic API key: ")
        #     self.model = ChatAnthropic(
        #         anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
        #         model=self.model_name,
        #         temperature=self.temperature,
        #         max_tokens=self.max_tokens
        #     )

        # elif self.llm_provider == "MISTRAL":
        #     # For Mistral
        #     if "MISTRAL_API_KEY" not in os.environ:
        #         os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter Mistral API key: ")
        #     self.model = ChatMistralAI(
        #         mistral_api_key=os.environ["MISTRAL_API_KEY"],
        #         model=self.model_name,
        #         temperature=self.temperature,
        #         max_tokens=self.max_tokens
        #     )

        # elif self.llm_provider == "GOOGLE":
        #     # For PaLM/Google Generative AI
        #     if "GOOGLE_API_KEY" not in os.environ:
        #         os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")
        #     self.model = ChatGoogleGenerativeAI(
        #         google_api_key=os.environ["GOOGLE_API_KEY"],
        #         model=self.model_name,
        #         temperature=self.temperature,
        #         max_tokens=self.max_tokens
        #     )

        elif self.llm_provider == "Custom":
            """
            This means: 
              - We have a local or remote OpenLLM server started with `openllm start ...`
              - We'll prompt for the server URL, e.g. http://localhost:3000
              - We'll pass it as openai_api_base to mimic an "OpenAI-compatible" endpoint.
              - We also pass a dummy openai_api_key to satisfy the pydantic schema.
            """
            server_url = input("Enter the server URL (e.g. http://localhost:3000): ")
            # In some versions, you might need to add "/v1" at the end to fully mimic OpenAI's format
            if not server_url.endswith("/v1"):
                server_url = server_url.rstrip("/") + "/v1" # some times they request to ../v1/chat or ../v1 so may need to edit here

            self.model = OpenLLM(
                openai_api_key="dummy_server_key",
                openai_api_base=server_url,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        else:
            raise ValueError(
                f"LLM provider '{self.llm_provider}' not supported. "
                "Choose from: OpenAI, ANTHROPIC, MISTRAL, GOOGLE, or Custom."
            )

        if self.model is None:
            raise RuntimeError(
                f"Failed to load LLM for provider '{self.llm_provider}'. "
                "Check your configuration or provider logic."
            )

    def get_model(self):
        """
        Return the loaded LangChain-compatible model 
        (ChatOpenAI, ChatAnthropic, ChatMistralAI, ChatGoogleGenerativeAI, or OpenLLM).
        """
        return self.model
