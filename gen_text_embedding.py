from uuid import uuid4
from langchain_core.documents import Document
import sys
import getpass
import os
from dotenv import load_dotenv
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# Check required environment variables
required_vars = [
    "AZURE_OPENAI_ENDPOINT", 
    "AZURE_OPENAI_DEPLOYMENT_NAME", 
    "AZURE_OPENAI_API_VERSION",
    "MONGODB_ATLAS_CLUSTER_URI"
]

missing_vars = [var for var in required_vars if not os.environ.get(var)]
if missing_vars:
    logging.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    sys.exit(1)

if not os.environ.get("AZURE_OPENAI_API_KEY"):
    os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Enter API key for Azure: ")

try:
    from langchain_openai import AzureOpenAIEmbeddings
    
    logging.info("Initializing Azure OpenAI embeddings...")
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )
    
    # Test embeddings with a simple string
    try:
        test_embedding = embeddings.embed_query("Test embedding")
        logging.info(f"Successfully generated test embedding (length: {len(test_embedding)})")
    except Exception as e:
        logging.error(f"Error generating test embedding: {e}")
        sys.exit(1)
        
except Exception as e:
    logging.error(f"Error initializing Azure OpenAI embeddings: {e}")
    sys.exit(1)

MONGODB_ATLAS_CLUSTER_URI = os.environ.get("MONGODB_ATLAS_CLUSTER_URI")

# Initialize MongoDB python client
try:
    logging.info("Connecting to MongoDB Atlas...")
    client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
    # Test connection
    client.admin.command('ping')
    logging.info("Successfully connected to MongoDB Atlas")
except Exception as e:
    logging.error(f"Error connecting to MongoDB Atlas: {e}")
    sys.exit(1)

DB_NAME = "styva-demo"
COLLECTION_NAME = "docs"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "docs-index-vectorstores"

MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

try:
    logging.info("Initializing MongoDB Atlas vector store...")
    vector_store = MongoDBAtlasVectorSearch.from_connection_string(
        connection_string=MONGODB_ATLAS_CLUSTER_URI,
        namespace=f"{DB_NAME}.{COLLECTION_NAME}",
        embedding=embeddings,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )
    logging.info("Successfully initialized vector store")
except Exception as e:
    logging.error(f"Error initializing vector store: {e}")
    sys.exit(1)

# Define documents
documents = [
    Document(
        page_content="I had chocalate chip pancakes and scrambled eggs for breakfast this morning.",
        metadata={"source": "tweet"},
    ),
    Document(
        page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
        metadata={"source": "news"},
    ),
    Document(
        page_content="Building an exciting new project with LangChain - come check it out!",
        metadata={"source": "tweet"},
    ),
    Document(
        page_content="Robbers broke into the city bank and stole $1 million in cash.",
        metadata={"source": "news"},
    ),
    Document(
        page_content="Wow! That was an amazing movie. I can't wait to see it again.",
        metadata={"source": "tweet"},
    ),
    Document(
        page_content="Is the new iPhone worth the price? Read this review to find out.",
        metadata={"source": "website"},
    ),
    Document(
        page_content="The top 10 soccer players in the world right now.",
        metadata={"source": "website"},
    ),
    Document(
        page_content="LangGraph is the best framework for building stateful, agentic applications!",
        metadata={"source": "tweet"},
    ),
    Document(
        page_content="The stock market is down 500 points today due to fears of a recession.",
        metadata={"source": "news"},
    ),
    Document(
        page_content="I have a bad feeling I am going to get deleted :(",
        metadata={"source": "tweet"},
    ),
]

# Generate UUIDs for documents
uuids = [str(uuid4()) for _ in range(len(documents))]

try:
    logging.info(f"Adding {len(documents)} documents to vector store...")
    vector_store.add_documents(documents=documents, ids=uuids)
    logging.info("Successfully added documents to vector store")
    
    # Test a simple query to verify everything worked
    query = "What's the weather like?"
    logging.info(f"Testing vector search with query: '{query}'")
    results = vector_store.similarity_search(query, k=2)
    logging.info(f"Search results: {[doc.page_content for doc in results]}")
    
except Exception as e:
    logging.error(f"Error adding documents to vector store: {e}")
    sys.exit(1)

logging.info("Script completed successfully")
