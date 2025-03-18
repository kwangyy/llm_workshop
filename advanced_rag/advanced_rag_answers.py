import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from openai import OpenAI
from huggingface_hub import InferenceClient
import re 
from typing import Optional, Dict, Any
import json
import pandas as pd

# Load API keys from .env file
load_dotenv()

# Initialize Pinecone instance
pc = Pinecone(os.getenv("PINECONE_API_KEY"))

# Define Pinecone index name
index_name = "rag-index"

# Check if the index exists; if not, create it.
existing_indexes = [index["name"] for index in pc.list_indexes()]

if index_name not in existing_indexes:
    print(f"Creating index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",  
            region="us-east-1"
        )
    )
else:
    print(f"Index '{index_name}' already exists. Skipping creation.")

# Initialize Pinecone index
index = pc.Index(index_name)

# Initialize OpenAI client
client = OpenAI()

# Initialize HuggingFace client
hugging_face_client = InferenceClient(api_key=os.getenv("HF_API_KEY"))

# Utility function to extract JSON from response text
def extract_json_from_response(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Extracts JSON from response text, handling code blocks and multiline content.
    Removes comments before parsing.
    """
    def remove_comments(json_str):
        json_str = re.sub(r'#.*$', '', json_str, flags=re.MULTILINE)
        json_str = re.sub(r'//.*$', '', json_str, flags=re.MULTILINE)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        return json_str

    # First try to find JSON within code blocks
    code_block_pattern = r'```json\s*([\s\S]*?)\s*```'
    code_block_match = re.search(code_block_pattern, response_text)
    
    if code_block_match:
        try:
            json_str = code_block_match.group(1)
            json_str = remove_comments(json_str)
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # If no code block or invalid JSON, try general JSON pattern
    json_pattern = r'(?s)\{.*?\}(?=\s*$)'
    json_match = re.search(json_pattern, response_text)
    
    if json_match:
        try:
            json_str = json_match.group(0)
            json_str = remove_comments(json_str)
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    return None

query = "What is the use of solar energy?"

# Expand the query using LLMs with JSON fields "query" and "expanded_query"
# Extract the json from the answer itself
rewritten_messages = [
    {
        "role": "system",
        "content": """You are an expert at rewriting queries. 
        You are to return a json with fields "query" and "expanded_query". 
        The "query" is the query from the user, and the "expanded_query" is the expanded query. 
        The expanded query should be more specific and to the point than the original query. 
        The expanded query should be in the form of a question, not a statement. 
        If the query is specific enough, simply return the query as the expanded query.
        Return JSON only, no other text.
        Examples of the output are: 
        {{
            "query": "Why is solar energy good?",
            "expanded_query": "What are the features of solar energy that make it more advantageous than other forms of energy?"
        }}
        {{
            "query": "What is the capital of France?",
            "expanded_query": "What is the capital of France?"
        }}
        {{
            "query": "How does the sun generate energy?",
            "expanded_query": "How is the sun used in technology to generate energy?"
        }}
        """
    },
    {
        "role": "user",
        "content": f"Rewrite the following query: {query}"
    }   
]
hugging_face_completion = hugging_face_client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct", 
    messages=rewritten_messages, 
    max_tokens=500,
)   
try:
    print(hugging_face_completion.choices[0].message.content)
    rewritten_json = extract_json_from_response(hugging_face_completion.choices[0].message.content)
    query = rewritten_json["expanded_query"]
except Exception as e:
    print(f"Error extracting JSON from response: {e}")


# Function to convert text into embeddings
def embed_text(text):
    embedding = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return embedding.data[0].embedding

# Function to split large text into smaller chunks
def chunk_text(text, chunk_size=40):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Read and chunk the energy.txt document
with open("energy.txt", "r", encoding="utf-8") as file:
    document_text = file.read()

chunks = chunk_text(document_text, chunk_size=40)  # Adjust chunk size as needed

# For each chunk, create more queries based on the chunk using LLMs
# Return the json with fields "content" and "queries"
# Extract the json from the answer itself
# Optional: If LLM is not able to extract the json, restart the LLM 
data = {}
count = 0 
for chunk in chunks:
    expanded_messages = [
        {
            "role": "system",
            "content": """You are an expert at asking more questions based on the given text. 
            You are to return a json with fields "content" and "queries". 
            The "content" is text from the user that is a chunk of a larger text, 
            and the "queries" are questions that you would ask in order to answer the "content" field. 
            The questions should be based on the content, and should be specific and to the point. 
            The questions should be in the form of a question, not a statement. 
            Return JSON only, no other text.
            An example of the output is: 
            {{
                "content": "Solar energy has been used for heating buildings for a long time. With the development of PV technology, converting solar radiation to electricity, photovoltaic systems seem to be more and more promising in providing a part of the total electricity demand.",
                "queries": [
                    "How has solar energy traditionally been used in buildings?",
                    "What technological advancement has improved the conversion of solar radiation into electricity?",
                    "Why do photovoltaic systems seem increasingly promising for meeting electricity demand?"
                ]
            }}"""
        },
        {
            "role": "user",
            "content": f"Write questions based on the following chunk of text: {chunk}"
        }
    ]
    hugging_face_completion = hugging_face_client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct", 
        messages=expanded_messages, 
        max_tokens=500,
    )
    try:
        print(hugging_face_completion.choices[0].message.content)
        json_response = extract_json_from_response(hugging_face_completion.choices[0].message.content)
        data[count] = json_response
        count += 1
    except Exception as e:
        print(f"Error extracting JSON from response: {e}")
        continue

for i in data.keys():
    try:
        print(f"Upserting vector {i}")
        for generated_query in data[i]["queries"]:
            # Embed the query
            vector = embed_text(generated_query)
            index.upsert(
                vectors=[
                    {
                        "id": str(i),
                        "values": vector,
                        "metadata": {"text": data[i]["content"],
                                     "query": generated_query}  # Store original text for retrieval
                    }
                    ]
                )
    except Exception as e:
        print(f"Error upserting vector {i}: {e}")

# Create query embedding
print(query)
query_embedding = embed_text(query)

# Query Pinecone index
response = index.query(
    vector=query_embedding,
    top_k=5,  # Retrieve the 5 most similar chunks
    include_metadata=True
)
print(response)
top_chunk = response["matches"][0]["metadata"]["text"]

print("Top chunk:", top_chunk)

# Simple test function
def test():
    print("RAG assignment is working!")

if __name__ == "__main__":
    test()
