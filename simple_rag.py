import os 
import pinecone 
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Initialize Pinecone instance
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
)
# Initialize Pinecone index
index = pinecone.Index("rag-index")

# Initialize OpenAI client
client = OpenAI()

# Embed text
def embed_text(text):
    # Embed the text string
    embedding = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return embedding.data[0].embedding

text = "Hello, world!"
text_embedding = embed_text(text)

print("Embedding:", text_embedding)

# Upsert the vectors into pinecone
pinecone.upsert(
    vectors=[
        {
            "id": "1",
            "values": text_embedding
        }
    ]
)

# Query Pinecone index
query = "Your query string goes here"
query_embedding = embed_text(query)

response = index.query(
    top_k=10,
    include_metadata=True,
    vector=query_embedding
)

print("Response:", response)

def test():
    print("Hello, world!")

if __name__ == "__main__":
    test()
