import getpass
import os
from dotenv import load_dotenv
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

# Load environment variables from .env file
load_dotenv()

if not os.environ.get("AZURE_OPENAI_API_KEY"):
  os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Enter API key for Azure: ")

from langchain_openai import AzureOpenAIEmbeddings

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)

MONGODB_ATLAS_CLUSTER_URI = os.environ.get("MONGODB_ATLAS_CLUSTER_URI")

# initialize MongoDB python client
client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)

DB_NAME = "styva-demo"
COLLECTION_NAME = "docs"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "docs-index-vectorstores"

MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

vector_store = MongoDBAtlasVectorSearch(
    collection=MONGODB_COLLECTION,
    embedding=embeddings,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    relevance_score_fn="cosine",
)

# Create vector search index on the collection
vector_store.create_vector_search_index(dimensions=1536)