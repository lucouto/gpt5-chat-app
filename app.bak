from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
import os
import json
import redis
from openai import AzureOpenAI

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
    # Try Redis first if enabled
    if redis_enabled and redis_client:
        try:
            cached = redis_client.get(f"conv:{conversation_id}")
            if cached:
                return json.loads(cached)
        except Exception as e:
            print(f"Redis get error: {e}")
    
    # Fallback to memory
    return conversations_memory.get(conversation_id)

def save_conversation(conversation_id, conversation_data):
    """Save conversation to Redis and memory"""
    # Always save to memory as backup
    conversations_memory[conversation_id] = conversation_data
    
    # Save to Redis if enabled
    if redis_enabled and redis_client:
        try:
            # Save with 7 days expiration (604800 seconds)
            redis_client.setex(
                f"conv:{conversation_id}",
                604800,
                json.dumps(conversation_data)
            )
            return True
        except Exception as e:
            print(f"Redis save error: {e}")
            return False
    return False

def delete_conversation(conversation_id):
    """Delete conversation from Redis and memory"""
    # Delete from memory
    if conversation_id in conversations_memory:
        del conversations_memory[conversation_id]
    
    # Delete from Redis if enabled
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
        
        # Get or create conversation
        conversation = get_conversation(conversation_id)
        if not conversation:
            conversation = create_new_conversation()
        
        # Add user message to conversation
        conversation.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_message
                }
            ]
        })
        
        # Generate completion from Azure OpenAI
        completion = client.chat.completions.create(
            model=deployment,
            messages=conversation,
            max_completion_tokens=16384,
            stop=None,
            stream=False
        )
        
        # Extract assistant response
        assistant_message = completion.choices[0].message.content
        
        # Add assistant response to conversation
        conversation.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": assistant_message
                }
            ]
        })
        
        # Save updated conversation
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
        
        # Delete conversation
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
    """Health check endpoint"""
    redis_status = "connected" if redis_enabled else "disabled"
    return jsonify({
        'status': 'healthy',
        'redis': redis_status,
        'azure_openai': 'configured' if subscription_key else 'not configured'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
