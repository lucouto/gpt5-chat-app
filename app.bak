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

# Authentication credentials (from environment variables)
users = {
    os.getenv("AUTH_USERNAME", "admin"): generate_password_hash(os.getenv("AUTH_PASSWORD", "changeme"))
}

@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username
    return None

# Initialize Azure OpenAI client
endpoint = os.getenv("ENDPOINT_URL", "https://ccn-openai-sweden.openai.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-5-chat")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

if not subscription_key:
    raise ValueError("AZURE_OPENAI_API_KEY environment variable is required")

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2025-01-01-preview",
)

# === RAG ADDITION: Azure Cognitive Search client ===
search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_key = os.getenv("AZURE_SEARCH_API_KEY")
search_index = os.getenv("AZURE_SEARCH_INDEX")
embedding_deployment = os.getenv("EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

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

# Initialize Redis client for persistent storage
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
        # Test connection
        redis_client.ping()
        redis_enabled = True
        print("✅ Redis connected successfully - Conversations will be persistent")
    except Exception as e:
        print(f"⚠️  Redis connection failed: {e}")
        print("📝 Falling back to in-memory storage (conversations will be lost on restart)")
        redis_client = None
        redis_enabled = False
else:
    print("📝 REDIS_URL not set - Using in-memory storage (conversations will be lost on restart)")

# In-memory backup storage (used when Redis is unavailable)
conversations_memory = {}

def get_conversation(conversation_id):
    """Retrieve conversation from Redis or memory"""
    if redis_enabled and redis_client:
        try:
            cached = redis_client.get(f"conv:{conversation_id}")
            if cached:
                return json.loads(cached)
        except Exception as e:
            print(f"Redis get error: {e}")
    return conversations_memory.get(conversation_id)

def save_conversation(conversation_id, conversation_data):
    """Save conversation to Redis and memory"""
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
    """Delete conversation from Redis and memory"""
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
    """Create a new conversation with system prompt"""
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

# === RAG ADDITION: Context retrieval ===
def retrieve_context(user_query):
    """Retrieve top matching chunks from Azure Cognitive Search"""
    if not search_client:
        return ""
    try:
        embedding = client.embeddings.create(
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

        chunks = []
        for doc in results:
            chunks.append(doc["chunk"])
        return "\n\n".join(chunks)
    except Exception as e:
        print(f"RAG retrieval error: {e}")
        return ""

@app.route('/')
@auth.login_required
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/chat', methods=['POST'])
@auth.login_required
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        conversation_id = data.get('conversation_id', 'default')
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        conversation = get_conversation(conversation_id)
        if not conversation:
            conversation = create_new_conversation()
        
        # === RAG ADDITION: Fetch context and inject into conversation ===
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
        
        completion = client.chat.completions.create(
            model=deployment,
            messages=conversation,
            max_completion_tokens=16384,
            stop=None,
            stream=False
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
        print(f"Error in chat endpoint: {str(e)}")
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
        print(f"Error in reset endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    redis_status = "connected" if redis_enabled else "disabled"
    return jsonify({
        'status': 'healthy',
        'redis': redis_status,
        'azure_openai': 'configured' if subscription_key else 'not configured',
        'azure_search': 'configured' if search_client else 'not configured'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
