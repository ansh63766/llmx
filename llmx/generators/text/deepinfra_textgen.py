import os
from typing import Union, List, Dict
from dataclasses import asdict, dataclass
import requests

from .base_textgen import TextGenerator
from ...datamodel import TextGenerationConfig, TextGenerationResponse
from ...utils import cache_request


@dataclass
class DialogueTemplate:
    system: str = None
    dialogue_type: str = "default"
    messages: list[dict[str, str]] = None
    system_token: str = "<|system|>"
    user_token: str = "<|user|>"
    assistant_token: str = "<|assistant|>"
    end_token: str = "<|end|>"

    def get_inference_prompt(self) -> str:
        if self.dialogue_type == "default":
            prompt = ""
            system_prompt = (
                self.system_token + "\n" + self.system + self.end_token + "\n"
                if self.system
                else ""
            )
            if self.messages is None:
                raise ValueError("Dialogue template must have at least one message.")
            for message in self.messages:
                if message["role"] == "system":
                    system_prompt += (
                        self.system_token
                        + "\n"
                        + message["content"]
                        + self.end_token
                        + "\n"
                    )
                elif message["role"] == "user":
                    prompt += (
                        self.user_token
                        + "\n"
                        + message["content"]
                        + self.end_token
                        + "\n"
                    )
                else:
                    prompt += (
                        self.assistant_token
                        + "\n"
                        + message["content"]
                        + self.end_token
                        + "\n"
                    )
            prompt += self.assistant_token
            if system_prompt:
                prompt = system_prompt + prompt
            return prompt
        else:
            raise NotImplementedError("Only default dialogue type is supported.")


class DeepInfraTextGenerator(TextGenerator):
    def __init__(self,
                 provider: str = "deepinfra",
                 endpoint_url: str = None,
                 api_key: str = None,
                 dialogue_type: str = "default",
                 **kwargs):
        super().__init__(provider=provider)

        if not endpoint_url or not api_key:
            raise ValueError("DeepInfra 'endpoint_url' and 'api_key' must be provided.")

        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.dialogue_template = DialogueTemplate(dialogue_type=dialogue_type)
        self.max_length = kwargs.get("max_length", 1024)

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text.
        You can replace this with a more accurate tokenization logic if necessary.
        For now, we split by spaces for simplicity.
        """
        return len(text.split())  # Simple tokenization by spaces. Adjust as needed.

    def post_process_response(self, response_text):
        response = (
            response_text.split(self.dialogue_template.assistant_token)[-1]
            .replace(self.dialogue_template.end_token, "")
            .strip()
        )
        return {"role": "assistant", "content": response}

    def generate(
            self, messages: Union[List[Dict], str],
            config: TextGenerationConfig = TextGenerationConfig(),
            **kwargs) -> TextGenerationResponse:
        use_cache = config.use_cache
        config.model = self.endpoint_url
        cache_key_params = {
            **asdict(config),
            **kwargs,
            "messages": messages,
            "dialogue_type": self.dialogue_template.dialogue_type
        }
        if use_cache:
            response = cache_request(cache=self.cache, params=cache_key_params)
            if response:
                return TextGenerationResponse(**response)

        # Construct the prompt from messages
        if isinstance(messages, list):
            prompt = ""
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                if role == "user":
                    prompt += f"<|user|>\n{content}<|end|>\n"
                elif role == "assistant":
                    prompt += f"<|assistant|>\n{content}<|end|>\n"
                else:
                    prompt += f"<|system|>\n{content}<|end|>\n"
        else:
            prompt = messages  # If a single string is passed as the prompt.

        # Prepare request payload
        payload = {
            "input": prompt,
            "parameters": {
                "max_new_tokens": kwargs.get("max_new_tokens", 200),
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": kwargs.get("top_k", config.top_k),
                "repetition_penalty": kwargs.get("repetition_penalty", 1.0),
            }
        }

        headers = {"Authorization": f"Bearer {self.api_key}"}

        # Make request to DeepInfra API
        response = requests.post(self.endpoint_url, json=payload, headers=headers)

        # Log the full response for debugging
        print(f"DeepInfra API Response Status Code: {response.status_code}")
        print(f"Response Content: {response.text}")

        if response.status_code != 200:
            raise ValueError(f"DeepInfra API request failed: {response.text}")

        # Process response
        response_json = response.json()
        generated_text = response_json.get("results", [{}])[0].get("generated_text", "")
        
        if not generated_text:
            print("Generated text is empty!")

        processed_response = self.post_process_response(generated_text)

        usage = {
            "prompt_tokens": self.count_tokens(prompt),
            "completion_tokens": self.count_tokens(generated_text) - self.count_tokens(prompt),
            "total_tokens": self.count_tokens(generated_text),
        }

        response = TextGenerationResponse(
            text=[processed_response],
            logprobs=[],
            config=config,
            usage=usage,
        )

        if use_cache:
            cache_request(cache=self.cache, params=cache_key_params, values=asdict(response))

        return response
