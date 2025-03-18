import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from openai import OpenAI

# Load API keys from .env file
load_dotenv()
# Initialize Pinecone

# Define Pinecone index name

# Check if the index exists; if not, create it.

# Connect to the existing index

# Initialize OpenAI client (for text embeddings)

# Function to convert text into embeddings

# Function to split large text into smaller chunks

# Read and chunk the energy.txt document

# Upsert each chunk into Pinecone

# Query Pinecone

# Display retrieved results

# Simple test function 