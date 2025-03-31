import os
import sys
import logging
import gradio as gr
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import AzureOpenAIEmbeddings

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Check required environment variables
required_vars = [
    "AZURE_OPENAI_ENDPOINT", 
    "AZURE_OPENAI_DEPLOYMENT_NAME", 
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_API_KEY",
    "MONGODB_ATLAS_CLUSTER_URI"
]

missing_vars = [var for var in required_vars if not os.environ.get(var)]
if missing_vars:
    logging.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    sys.exit(1)

# Initialize Azure OpenAI Embeddings
try:
    logging.info("Initializing Azure OpenAI embeddings...")
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )
    logging.info("Successfully initialized embeddings")
except Exception as e:
    logging.error(f"Error initializing Azure OpenAI embeddings: {e}")
    sys.exit(1)

# Initialize MongoDB connection
try:
    logging.info("Connecting to MongoDB Atlas...")
    client = MongoClient(os.environ["MONGODB_ATLAS_CLUSTER_URI"])
    # Test connection
    client.admin.command('ping')
    logging.info("Successfully connected to MongoDB Atlas")
except Exception as e:
    logging.error(f"Error connecting to MongoDB Atlas: {e}")
    sys.exit(1)

# Setup Vector Store
DB_NAME = "styva-demo"
COLLECTION_NAME = "docs"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "docs-index-vectorstores"

try:
    logging.info("Initializing MongoDB Atlas vector store...")
    vector_store = MongoDBAtlasVectorSearch.from_connection_string(
        connection_string=os.environ["MONGODB_ATLAS_CLUSTER_URI"],
        namespace=f"{DB_NAME}.{COLLECTION_NAME}",
        embedding=embeddings,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )
    logging.info("Successfully initialized vector store")
except Exception as e:
    logging.error(f"Error initializing vector store: {e}")
    sys.exit(1)

def search_documents(query, chat_history, num_results=3):
    """
    Search for documents in the vector store based on the query
    """
    try:
        if not query.strip():
            return chat_history + [[query, "Please enter a valid search query"]]
            
        logging.info(f"Searching for: '{query}'")
        results = vector_store.similarity_search(query, k=num_results)
        
        if not results:
            return chat_history + [[query, "I couldn't find any relevant information for your query."]]
            
        response = "Here's what I found:\n\n"
        for i, doc in enumerate(results, 1):
            response += f"**Document {i}**\n"
            response += f"**Content:** {doc.page_content}\n"
            response += f"**Source:** {doc.metadata.get('source', 'Unknown')}\n\n"
        
        return chat_history + [[query, response]]
    except Exception as e:
        logging.error(f"Error during search: {e}")
        error_message = f"I encountered an error while searching: {str(e)}"
        return chat_history + [[query, error_message]]

# Create Gradio chatbot interface
with gr.Blocks(title="Vector Search Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Vector Search Chatbot")
    gr.Markdown("Ask me questions about the documents in the database and I'll find relevant information.")
    
    chatbot = gr.Chatbot(
        height=500,
        show_label=False,
        show_share_button=False,
        show_copy_button=True,
        avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=search-bot")
    )
    
    with gr.Row():
        query_input = gr.Textbox(
            placeholder="Ask me something...", 
            show_label=False,
            lines=2,
            container=False
        )
        submit_button = gr.Button("Search", variant="primary")
    
    num_results_slider = gr.Slider(
        minimum=1, 
        maximum=10, 
        value=3, 
        step=1, 
        label="Number of Results to Return"
    )
    
    gr.Examples(
        examples=[
            "Stealing from the bank is a crime",
            "Will it be hot tomorrow?", 
        ],
        inputs=query_input
    )
    
    # Welcome message when the app starts
    def get_welcome_message():
        return [[None, "👋 Hello! I'm your vector search assistant. Ask me something about the documents in the database!"]]
    
    # Set initial state with welcome message
    chatbot.value = get_welcome_message()
    
    # Handle form submission
    def on_submit(message, chat_history, num_results):
        if message.strip() == "":
            return "", chat_history
        return "", search_documents(message, chat_history, num_results)
    
    submit_button.click(
        on_submit, 
        inputs=[query_input, chatbot, num_results_slider], 
        outputs=[query_input, chatbot]
    )
    
    query_input.submit(
        on_submit, 
        inputs=[query_input, chatbot, num_results_slider], 
        outputs=[query_input, chatbot]
    )
    
    # Add clear button
    clear_button = gr.Button("Clear Chat")
    clear_button.click(
        lambda: get_welcome_message(),
        outputs=[chatbot]
    )

if __name__ == "__main__":
    logging.info("Starting Gradio chatbot application...")
    demo.launch(share=True)
    logging.info("Gradio application closed")