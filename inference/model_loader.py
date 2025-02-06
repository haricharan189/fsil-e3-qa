import os
import getpass
import requests
import openai

# For open-source LLM usage (OpenLLM)
from langchain_community.llms import OpenLLM

# For proprietary LLM usage
from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
# from langchain_mistralai import ChatMistralAI
# from langchain_google_genai import ChatGoogleGenerativeAI

from vllm import LLM, SamplingParams

class Response:
    def __init__(self, content):
        self.content = content


class ChatTogether:
    """
    Minimal wrapper for TogetherAI endpoint that mimics the LangChain chat interface.
    Provides an 'invoke(messages)' method returning an object with a .content attribute.
    """
    def __init__(self, together_api_key, model, temperature=0.7, max_tokens=2000):
        self.api_url = "https://api.together.xyz/v1/chat/completions"
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {together_api_key}"
        }
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, messages):
        """
        Makes a POST request to TogetherAI with the given messages.
        Each item in messages is expected to be a dict with 'role' and 'content'.
        Returns an object with a `content` attribute (like ChatOpenAI's response).
        """
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "messages": messages
        }

        resp = requests.post(self.api_url, json=payload, headers=self.headers)
        if resp.status_code != 200:
            raise ValueError(f"[TogetherAI] Error {resp.status_code}: {resp.text}")

        data = resp.json()
        # The structure is: data["choices"][0]["message"]["content"] for the text
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        return Response(content)


class ChatOpenSource:
    def __init__(self, model_name, temperature):
        self.model_name = model_name
        self.sampling_params = SamplingParams(temperature=temperature)
        self.model = LLM(model=self.model_name)

    def invoke(self, messages):
        prompts = [message['content'] for message in messages]
        outputs = self.model.generate(prompts, self.sampling_params)
        content = "\n".join(output.outputs[0].text for output in outputs)
        return Response(content)


class BaseModel:
    """
    Loads an LLM based on the specified provider (OpenAI, Anthropic, Mistral, Google,
    Together, or OpenLLM via local or custom server).
    """

    def __init__(self, llm_provider, model_name, temperature, max_tokens):
        """
        :param llm_provider: e.g. "OpenAI", "ANTHROPIC", "MISTRAL", "GOOGLE", "TOGETHER", or "Custom"
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

        elif self.llm_provider == "TOGETHER":
            # For TogetherAI (open-source models)
            if "TOGETHER_API_KEY" not in os.environ:
                os.environ["TOGETHER_API_KEY"] = getpass.getpass("Enter Together API key: ")
            self.model = ChatTogether(
                together_api_key=os.environ["TOGETHER_API_KEY"],
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

        elif self.llm_provider == "Custom":
            """
            This means: 
              - We have a local or remote OpenLLM server started with `openllm start ...`
              - We'll prompt for the server URL, e.g. http://localhost:3000
              - We'll pass it as openai_api_base to mimic an "OpenAI-compatible" endpoint.
              - We also pass a dummy openai_api_key to satisfy the pydantic schema.
            """
            self.model = ChatOpenSource(
                model_name=self.model_name,
                temperature=self.temperature
            )
        else:
            raise ValueError(
                f"LLM provider '{self.llm_provider}' not supported. "
                "Choose from: OpenAI, ANTHROPIC, MISTRAL, GOOGLE, TOGETHER, or Custom."
            )

        if self.model is None:
            raise RuntimeError(
                f"Failed to load LLM for provider '{self.llm_provider}'. "
                "Check your configuration or provider logic."
            )

    def get_model(self):
        """
        Return the loaded model that exposes `invoke(messages)`.
        """
        return self.model
