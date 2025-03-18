import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from openai import OpenAI

# Load API keys from .env file
load_dotenv()

# Initialize Pinecone
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

# Connect to the existing index
index = pc.Index(index_name)

# Initialize OpenAI client (for text embeddings)
client = OpenAI()

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

# Upsert each chunk into Pinecone
for i, chunk in enumerate(chunks):
    vector = embed_text(chunk)  # Convert chunk to an embedding
    index.upsert(
        vectors=[
            {
                "id": str(i),
                "values": vector,
                "metadata": {"text": chunk}  # Store original text for retrieval
            }
        ]
    )

# Query Pinecone
query = "Who sourced this article?"
query_embedding = embed_text(query)

response = index.query(
    vector=query_embedding,
    top_k=5,  # Retrieve the 5 most similar chunks
    include_metadata=True
)

# Display retrieved results
print("\nRetrieved Chunks:")
for match in response["matches"]:
    print(f"- Score: {match['score']:.4f} | Text: {match['metadata']['text'][:500]}...")  # Show first 100 characters

# Simple test function
def test():
    print("RAG assignment is working!")

if __name__ == "__main__":
    test()
