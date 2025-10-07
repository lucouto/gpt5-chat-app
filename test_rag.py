import os
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

# =========================
# Environment variable check
# =========================
required_env_vars = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_SEARCH_ENDPOINT",
    "AZURE_SEARCH_API_KEY",
    "AZURE_SEARCH_INDEX",
    "EMBEDDING_DEPLOYMENT"
]

missing = [var for var in required_env_vars if not os.getenv(var)]
if missing:
    raise EnvironmentError(f"Missing environment variables: {', '.join(missing)}")

# =========================
# Config from environment
# =========================
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")
EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT")  # e.g. "text-embedding-3-small"

# =========================
# Azure OpenAI client
# =========================
openai_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2025-01-01-preview"
)

# =========================
# Azure Search client
# =========================
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
)

# =========================
# RAG Search
# =========================
def rag_search(query):
    # Step 1: Get embedding from Azure OpenAI
    embedding = openai_client.embeddings.create(
        model=EMBEDDING_DEPLOYMENT,
        input=query
    ).data[0].embedding

    # Step 2: Search Azure Cognitive Search using vector search
    results = search_client.search(
        search_text=None,
        vector_queries=[{
            "kind": "vector",           # REQUIRED in latest API
            "vector": embedding,
            "fields": "text_vector",    # Name of your vector field in index
            "k": 3                      # Top K results
        }],
        select=["chunk", "title"]
    )

    # Step 3: Print results
    print("\n=== Search Results ===")
    found_any = False
    for doc in results:
        found_any = True
        title = doc.get("title", "No title")
        chunk = doc.get("chunk", "No chunk")
        print(f"Title: {title}")
        print(f"Chunk: {chunk}")
        print("----------------------")
    if not found_any:
        print("No results found.")

# =========================
# Main
# =========================
if __name__ == "__main__":
    user_query = input("Enter your test question: ")
    rag_search(user_query)
