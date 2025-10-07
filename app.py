from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
import os
import json
import redis
from openai import AzureOpenAI

# === RAG ADDITION ===
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

app = Flask(__name__, static_folder='static')
CORS(app)
auth = HTTPBasicAuth()

# Authentication credentials
users = {
    os.getenv("AUTH_USERNAME", "admin"): generate_password_hash(os.getenv("AUTH_PASSWORD", "changeme"))
}

@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username
    return None

# === Azure OpenAI Clients ===
# Chat client for GPT-5 completions
chat_client = AzureOpenAI(
    azure_endpoint=os.getenv("ENDPOINT_URL"),               # ccnsweden endpoint
    api_key=os.getenv("AZURE_OPENAI_API_KEY_CHAT"),         # chat key
    api_version="2025-01-01-preview",
)

# Embedding client for RAG vector search
embedding_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),      # openai-cnjeunes endpoint
    api_key=os.getenv("AZURE_OPENAI_API_KEY_EMBED"),        # embedding key
    api_version="2025-01-01-preview",
)

deployment = os.getenv("DEPLOYMENT_NAME", "gpt-5-chat")
embedding_deployment = os.getenv("EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

# === Azure Cognitive Search Client ===
search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_key      = os.getenv("AZURE_SEARCH_API_KEY")
search_index    = os.getenv("AZURE_SEARCH_INDEX")

search_client = None
if search_endpoint and search_key and search_index:
    try:
        search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=search_index,
            credential=AzureKeyCredential(search_key)
        )
        print("✅ Azure Cognitive Search connected for RAG")
    except Exception as e:
        print(f"⚠️ Azure Search connection failed: {e}")

# === Redis Client ===
redis_client = None
redis_enabled = False

if os.getenv("REDIS_URL"):
    try:
        redis_client = redis.from_url(
            os.getenv("REDIS_URL"),
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5
        )
        redis_client.ping()
        redis_enabled = True
        print("✅ Redis connected successfully - Conversations will be persistent")
    except Exception as e:
        print(f"⚠️ connection failed: {e}")
        redis_client = None
        redis_enabled = False
else:
    print("📝 REDIS_URL not set - Using in-memory storage")

# In-memory backup
conversations_memory = {}

def get_conversation(conversation_id):
    if redis_enabled and redis_client:
        try:
            cached = redis_client.get(f"conv:{conversation_id}")
            if cached:
                return json.loads(cached)
        except Exception as e:
            print(f"Redis get error: {e}")
    return conversations_memory.get(conversation_id)

def save_conversation(conversation_id, conversation_data):
    conversations_memory[conversation_id] = conversation_data
    if redis_enabled and redis_client:
        try:
            redis_client.setex(f"conv:{conversation_id}", 604800, json.dumps(conversation_data))
            return True
        except Exception as e:
            print(f"Redis save error: {e}")
            return False
    return False

def delete_conversation(conversation_id):
    if conversation_id in conversations_memory:
        del conversations_memory[conversation_id]
    if redis_enabled and redis_client:
        try:
            redis_client.delete(f"conv:{conversation_id}")
            return True
        except Exception as e:
            print(f"Redis delete error: {e}")
            return False
    return False

def create_new_conversation():
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are an AI assistant that helps people find information."
                }
            ]
        }
    ]

# === RAG Context Retrieval (for chat) ===
def retrieve_context(user_query):
    if not search_client:
        return ""
    try:
        embedding = embedding_client.embeddings.create(
            model=embedding_deployment,
            input=user_query
        ).data[0].embedding

        results = search_client.search(
            search_text=None,
            vector_queries=[{
                "kind": "vector",
                "vector": embedding,
                "fields": "text_vector",
                "k": 3
            }],
            select=["chunk", "title"]
        )

        chunks = [doc["chunk"] for doc in results]
        return "\n\n".join(chunks)
    except Exception as e:
        print(f"RAG retrieval error: {e}")
        return ""

# === NEW: RAG LOOKUP TOOL ENDPOINT ===
@app.route('/api/search', methods=['POST'])
@auth.login_required
def vector_search():
    """
    POST { "query": "..." }
    → 200 { "results":[{"title":"…","chunk":"…"},…] }
    → 400 if missing query
    → 500 if not configured or on error
    """
    data = request.get_json(silent=True) or {}
    query = data.get('query', '').strip()
    if not query:
        return jsonify({'error': 'query is required'}), 400

    if not search_client:
        return jsonify({'error': 'RAG search is not configured'}), 500

    try:
        # 1) Get embedding
        emb = embedding_client.embeddings.create(
            model=embedding_deployment,
            input=query
        ).data[0].embedding

        # 2) Vector search
        results = search_client.search(
            search_text=None,
            vector_queries=[{
                "kind": "vector",
                "vector": emb,
                "fields": "text_vector",
                "k": 3
            }],
            select=["title", "chunk"]
        )

        hits = [
            {"title": doc.get("title", ""), "chunk": doc.get("chunk", "")}
            for doc in results
        ]
        return jsonify({'results': hits}), 200

    except Exception as e:
        print(f"RAG search error: {e}")
        return jsonify({'error': str(e)}), 500

# === EXISTING UI / CHAT ENDPOINTS ===
@app.route('/')
@auth.login_required
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/chat', methods=['POST'])
@auth.login_required
def chat():
    try:
        data = request.json
        user_message    = data.get('message', '')
        conversation_id = data.get('conversation_id', 'default')
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        conversation = get_conversation(conversation_id) or create_new_conversation()

        # Inject RAG context
        context = retrieve_context(user_message)
        if context:
            conversation.append({
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"Relevant context from documents:\n{context}"
                    }
                ]
            })

        conversation.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_message
                }
            ]
        })
        
        completion = chat_client.chat.completions.create(
            model=deployment,
            messages=conversation,
            max_completion_tokens=16384
        )
        
        assistant_message = completion.choices[0].message.content
        
        conversation.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": assistant_message
                }
            ]
        })
        
        saved_to_redis = save_conversation(conversation_id, conversation)
        
        return jsonify({
            'response': assistant_message,
            'conversation_id': conversation_id,
            'storage': 'redis' if saved_to_redis else 'memory'
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
@auth.login_required
def reset_conversation():
    try:
        data = request.json
        conversation_id = data.get('conversation_id', 'default')
        deleted = delete_conversation(conversation_id)
        return jsonify({
            'message': 'Conversation reset successfully',
            'storage': 'redis' if deleted else 'memory'
        })
    except Exception as e:
        print(f"Error in reset endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    redis_status = "connected" if redis_enabled else "disabled"
    return jsonify({
        'status': 'healthy',
        'redis': redis_status,
        'azure_openai': 'configured' if os.getenv("AZURE_OPENAI_API_KEY_CHAT") else 'not configured',
        'azure_search': 'configured' if search_client else 'not configured'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
