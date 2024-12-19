import pytest
from llmx import llm
from llmx.datamodel import TextGenerationConfig

DEEPINFRA_API_KEY = "E8rm8LHXCd99PoSrQB2nh1T8Nb1ztkxw"  # API key
DEEPINFRA_ENDPOINT_URL = "https://api.deepinfra.com/v1/inference/meta-llama/Llama-2-13b-chat-hf"  # Endpoint URL

config = TextGenerationConfig(
    n=2,
    temperature=0.4,
    max_tokens=100,
    top_p=1.0,
    top_k=50,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    use_cache=False
)

messages = [
    {"role": "user",
     "content": "What is the capital of France? Only respond with the exact answer"}
]

def test_deepinfra():
    deepinfra_gen = llm(provider="deepinfra",
                        api_key=DEEPINFRA_API_KEY,
                        endpoint_url=DEEPINFRA_ENDPOINT_URL)
    
    # config.model = "Llama-2-13b-chat-hf"  # The actual model name used in the endpoint
    
    deepinfra_response = deepinfra_gen.generate(messages, config=config)
    
    answer = deepinfra_response.text[0].content
    print(answer)
    
    assert ("paris" in answer.lower())
