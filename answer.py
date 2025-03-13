import os 
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from simple_rag import test

load_dotenv()


client = InferenceClient(
    api_key=os.getenv("HF_API_KEY"),
)

messages = [
	{
		"role": "user",
		"content": "Who sourced this article?"
	}
]

completion = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct", 
	messages=messages, 
	max_tokens=500,
)

print(completion.choices[0].message.content)




