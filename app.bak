from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from openai import AzureOpenAI

app = Flask(__name__, static_folder='static')
CORS(app)

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

# Store conversation history (in production, use a proper database)
conversations = {}

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        conversation_id = data.get('conversation_id', 'default')
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Initialize conversation if it doesn't exist
        if conversation_id not in conversations:
            conversations[conversation_id] = [
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
        
        # Add user message to conversation
        conversations[conversation_id].append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_message
                }
            ]
        })
        
        # Generate completion
        completion = client.chat.completions.create(
            model=deployment,
            messages=conversations[conversation_id],
            max_completion_tokens=16384,
            stop=None,
            stream=False
        )
        
        # Extract assistant response
        assistant_message = completion.choices[0].message.content
        
        # Add assistant response to conversation
        conversations[conversation_id].append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": assistant_message
                }
            ]
        })
        
        return jsonify({
            'response': assistant_message,
            'conversation_id': conversation_id
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_conversation():
    try:
        data = request.json
        conversation_id = data.get('conversation_id', 'default')
        
        if conversation_id in conversations:
            del conversations[conversation_id]
        
        return jsonify({'message': 'Conversation reset successfully'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
