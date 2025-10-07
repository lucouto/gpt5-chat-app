import os
import json
import redis

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash

from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

# =========================
# REQUIRED ENVIRONMENT VARS
# =========================
REQUIRED = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_CHAT_DEPLOYMENT",     # e.g. your chat model deployment name
    "AZURE_EMBEDDING_DEPLOYMENT"        # e.g. text-embedding-3-small
]
missing = [v for v in REQUIRED if not os.getenv(v)]
if missing:
    raise EnvironmentError(f"Missing environment variables: {', '.join(missing)}")

# Optional Search vars (all three must be set to enable RAG)
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY  = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX    = os.getenv("AZURE_SEARCH_INDEX")

# Optional Redis
REDIS_URL = os.getenv("REDIS_URL")

# Basic-Auth
AUTH_USERNAME = os.getenv("AUTH_USERNAME", "admin")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "changeme")

# Extract mandatory vars
AZURE_OPENAI_ENDPOINT          = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY           = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_CHAT_DEPLOYMENT   = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
AZURE_EMBEDDING_DEPLOYMENT     = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

# =============
# FLASK SETUP
# =============
app = Flask(__name__, static_folder="static")
CORS(app)
auth = HTTPBasicAuth()

# =============
# BASIC AUTH
# =============
users = {
    AUTH_USERNAME: generate_password_hash(AUTH_PASSWORD)
}

@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users[username], password):
        return username
    return None

# ======================
# AZURE OPENAI CLIENT
# ======================
openai_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2025-01-01-preview",
)
print("✅ Azure OpenAI client initialized")

# ====================================
# AZURE COGNITIVE SEARCH (optional)
# ====================================
search_client = None
if AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY and AZURE_SEARCH_INDEX:
    try:
        search_client = SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=AZURE_SEARCH_INDEX,
            credential=AzureKeyCredential(AZURE_SEARCH_API_KEY),
        )
        print("✅ Azure Cognitive Search client initialized")
    except Exception as e:
        print(f"⚠️ Azure Search init failed: {e}")
else:
    print("📝 Azure Search not configured; /api/search will return 500")

# ======================
# REDIS (optional)
# ======================
redis_client = None
redis_enabled = False
if REDIS_URL:
    try:
        redis_client = redis.from_url(
            REDIS_URL, decode_responses=True,
            socket_timeout=5, socket_connect_timeout=5
        )
        redis_client.ping()
        redis_enabled = True
        print("✅ Redis initialized")
    except Exception as e:
        print(f"⚠️ Redis init failed: {e}")
else:
    print("📝 REDIS_URL not set; conversations will live in memory")

# In-memory fallback
conversations_memory = {}

def get_conversation(conv_id):
    if redis_enabled:
        try:
            data = redis_client.get(f"conv:{conv_id}")
            if data:
                return json.loads(data)
        except Exception:
            pass
    return conversations_memory.get(conv_id)

def save_conversation(conv_id, conv_data):
    conversations_memory[conv_id] = conv_data
    if redis_enabled:
        try:
            redis_client.setex(f"conv:{conv_id}", 604_800, json.dumps(conv_data))
            return True
        except Exception:
            pass
    return False

def delete_conversation(conv_id):
    conversations_memory.pop(conv_id, None)
    if redis_enabled:
        try:
            redis_client.delete(f"conv:{conv_id}")
            return True
        except Exception:
            pass
    return False

def create_new_conversation():
    return [
        {
            "role": "system",
            "content": [
                { "type": "text",
                  "text": "You are an AI assistant that helps people find information." }
            ]
        }
    ]

# =====================================
# CONTEXT RETRIEVAL HELPER (chat use)
# =====================================
def retrieve_context(user_query: str) -> str:
    """Return a big text blob of the top-3 document chunks."""
    if not search_client:
        return ""
    try:
        emb = openai_client.embeddings.create(
            model=AZURE_EMBEDDING_DEPLOYMENT,
            input=user_query
        ).data[0].embedding

        docs = search_client.search(
            search_text=None,
            vector_queries=[{
                "kind":   "vector",
                "vector": emb,
                "fields": "text_vector",
                "k":      3
            }],
            select=["title", "chunk"]
        )
        chunks = [d["chunk"] for d in docs]
        return "\n\n".join(chunks)
    except Exception as e:
        app.logger.error(f"RAG retrieval error: {e}")
        return ""

# =====================================
# NEW: VECTOR‐SEARCH “TOOL” ENDPOINT
# =====================================
@app.route("/api/search", methods=["POST"])
@auth.login_required
def api_search():
    """
    POST { "query": "..." }
    → 200 { "results": [ { title, chunk }, ... ] }
    → 400 if missing query
    → 500 if search not configured or on error
    """
    body = request.get_json(silent=True) or {}
    q = body.get("query", "").strip()
    if not q:
        return jsonify({"error": "query is required"}), 400

    if not search_client:
        return jsonify({"error": "RAG search is not configured"}), 500

    try:
        emb = openai_client.embeddings.create(
            model=AZURE_EMBEDDING_DEPLOYMENT,
            input=q
        ).data[0].embedding

        docs = search_client.search(
            search_text=None,
            vector_queries=[{
                "kind":   "vector",
                "vector": emb,
                "fields": "text_vector",
                "k":      3
            }],
            select=["title", "chunk"]
        )
        hits = [
            {"title": d.get("title",""), "chunk": d.get("chunk","")}
            for d in docs
        ]
        return jsonify({"results": hits}), 200

    except Exception as e:
        app.logger.error(f"RAG search error: {e}")
        return jsonify({"error": str(e)}), 500

# =====================================
# EXISTING: CHAT UI ENDPOINTS
# =====================================
@app.route("/")
@auth.login_required
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/chat", methods=["POST"])
@auth.login_required
def chat():
    body = request.get_json(silent=True) or {}
    user_msg = body.get("message", "").strip()
    conv_id  = body.get("conversation_id", "default")

    if not user_msg:
        return jsonify({"error": "message is required"}), 400

    conv = get_conversation(conv_id) or create_new_conversation()

    # Inject RAG context
    ctx = retrieve_context(user_msg)
    if ctx:
        conv.append({
            "role": "system",
            "content": [{ "type": "text",
                          "text": f"Relevant context from documents:\n{ctx}" }]
        })

    # Add user message
    conv.append({
        "role": "user",
        "content": [{ "type": "text", "text": user_msg }]
    })

    # Ask Azure OpenAI
    try:
        completion = openai_client.chat.completions.create(
            model=AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=conv,
            max_completion_tokens=4_096,
            stop=None,
            stream=False
        )
        assistant_msg = completion.choices[0].message.content
    except Exception as e:
        app.logger.error(f"Chat completion error: {e}")
        return jsonify({"error": str(e)}), 500

    # Save to history
    conv.append({
        "role": "assistant",
        "content": [{ "type": "text", "text": assistant_msg }]
    })
    saved = save_conversation(conv_id, conv)

    return jsonify({
        "response": assistant_msg,
        "conversation_id": conv_id,
        "storage": "redis" if saved else "memory"
    })

@app.route("/api/reset", methods=["POST"])
@auth.login_required
def reset_conv():
    body = request.get_json(silent=True) or {}
    conv_id = body.get("conversation_id", "default")
    deleted = delete_conversation(conv_id)
    return jsonify({
        "message": "Conversation reset",
        "storage": "redis" if deleted else "memory"
    })

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":      "healthy",
        "redis":       "connected"    if redis_enabled    else "disabled",
        "azure_openai":"configured"   if AZURE_OPENAI_API_KEY else "not configured",
        "azure_search":"configured"   if search_client     else "not configured"
    })

# ===========
# MAIN
# ===========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
