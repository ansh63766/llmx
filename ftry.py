# test_deepinfra.py
from llmx import llm

# Set your DeepInfra API key and endpoint URL
DEEPINFRA_API_KEY = "E8rm8LHXCd99PoSrQB2nh1T8Nb1ztkxw"  # API key
DEEPINFRA_ENDPOINT_URL = "https://api.deepinfra.com/v1/inference/meta-llama/Llama-2-13b-chat-hf"  # Endpoint URL


# Create the DeepInfra generator
gen = llm(provider="deepinfra", api_key=DEEPINFRA_API_KEY, endpoint_url=DEEPINFRA_ENDPOINT_URL)

# Define your conversation/messages
messages = [
    {"role": "user", "content": "Name any 3 states of India"}
]

# Generate a response
response = gen.generate(messages=messages)

# Print the output
print(response.text[0].content)
