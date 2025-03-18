import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from openai import OpenAI

# Load API keys
load_dotenv()

# Initialize Pinecone
pc = Pinecone(os.getenv("PINECONE_API_KEY"))

# Define Pinecone index
index_name = "rag-index"

# Ensure the index exists
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

# Connect to the existing index
index = pc.Index(index_name)

# Initialize OpenAI client
client = OpenAI()

# Function to convert text into embeddings
def embed_text(text):
    embedding = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return embedding.data[0].embedding

text = "Hello, world!"
text_embedding = embed_text(text)

print("Embedding:", text_embedding)

# Upsert the text into Pinecone
index.upsert(
    vectors=[
        {
            "id": "1",
            "values": text_embedding,
            "metadata": {"text": text}  # Store original text
        }
    ]
)

# Query Pinecone index
query = "Your query string goes here"
query_embedding = embed_text(query)

response = index.query(
    top_k=3,
    include_metadata=True,
    vector=query_embedding
)

print("Response:", response)

# Simple test function
def test():
    print("Simple RAG is working!")

if __name__ == "__main__":
    test()