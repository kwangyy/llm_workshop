import os 
import pinecone 
from dotenv import load_dotenv
from openai import OpenAI
from huggingface_hub import InferenceClient

load_dotenv()

# Initialize Pinecone instance

# Initialize Pinecone index

# Initialize OpenAI client

# Initialize HuggingFace client

# Expand the query using LLMs with JSON fields "query" and "expanded_query"
# Extract the json from the answer itself
# Optional: If LLM is not able to extract the json, restart the LLM 

# Chunk the text into smaller chunks

# For each chunk, create more queries based on the chunk using LLMs
# Return the json with fields "content" and "queries"

# Extract the json from the answer itself
# Optional: If LLM is not able to extract the json, restart the LLM 

# Embed the queries and upsert the remaining metadata into pinecone

# Create query embedding

# Query Pinecone index


