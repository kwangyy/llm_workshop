import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from openai import OpenAI
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(os.getenv("PINECONE_API_KEY"))

# Define Pinecone index
index_name = "advanced-rag-index"

# Ensure the index exists
existing_indexes = [index["name"] for index in pc.list_indexes()]

if index_name not in existing_indexes:
    print(f"Creating index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
else:
    print(f"Index '{index_name}' already exists. Skipping creation.")

# Connect to Pinecone index
index = pc.Index(index_name)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Hugging Face for query expansion
hf_client = InferenceClient(api_key=os.getenv("HF_API_KEY"))

# Function to embed text
def embed_text(text):
    embedding = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return embedding.data[0].embedding

# Function to expand queries using Hugging Face
def expand_query(query):
    response = hf_client.chat_completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[{"role": "user", "content": f"Expand this query: {query}"}]
    )
    return response["choices"][0]["message"]["content"]

# Function to rewrite queries using OpenAI
def rewrite_query(query):
    response = client.completions.create(
        model="gpt-4",
        prompt=f"Rewrite the following query for better retrieval:\n\n{query}",
        max_tokens=50
    )
    return response["choices"][0]["text"].strip()

# Function to retrieve documents with query expansion
def retrieve_text(query):
    expanded_query = expand_query(query)
    rewritten_query = rewrite_query(expanded_query)
    
    query_embedding = embed_text(rewritten_query)
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    
    return [match["metadata"]["text"] for match in results["matches"]]

# Example usage
if __name__ == "__main__":
    text_data = "This document explains Pinecone, OpenAI, and advanced LLM techniques."
    embed_text(text_data)
    
    query = "Tell me about Pinecone?"
    retrieved_docs = retrieve_text(query)
    
    print("Retrieved Documents:", retrieved_docs)