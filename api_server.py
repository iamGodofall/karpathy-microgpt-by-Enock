"""
REST API server for microgpt.
Provides HTTP endpoints for generation, training, and model management.
"""

import json
import random
from flask import Flask, request, jsonify
from model import GPT
from data import CharTokenizer
from checkpoint import CheckpointManager
from chat import ChatBot


app = Flask(__name__)

# Global model storage
models = {}
active_model = None
tokenizer = None


def load_model(checkpoint_path: str, model_name: str = "default"):
    """Load a model into the API."""
    global active_model, tokenizer
    
    checkpoint_mgr = CheckpointManager()
    checkpoint = checkpoint_mgr.load_pickle(checkpoint_path)
    
    config = checkpoint['config']
    
    model = GPT(
        vocab_size=config['model']['vocab_size'],
        block_size=config['model']['block_size'],
        n_layer=config['model']['n_layer'],
        n_embd=config['model']['n_embd'],
        n_head=config['model']['n_head'],
        use_gelu=config['model'].get('use_gelu', False),
        use_layernorm=config['model'].get('use_layernorm', False)
    )
    
    # Load weights
    for name, matrix_data in checkpoint['state_dict'].items():
        for i, row in enumerate(matrix_data):
            for j, val in enumerate(row):
                model.state_dict[name][i][j].data = val
    
    model.set_training(False)
    
    models[model_name] = model
    active_model = model
    
    # Create tokenizer
    tokenizer = CharTokenizer()
    tokenizer.fit(["example"])  # Would load actual vocab
    
    return model


@app.route('/')
def index():
    """API info endpoint."""
    return jsonify({
        'name': 'microgpt API',
        'version': '1.0.0',
        'endpoints': [
            '/generate',
            '/chat',
            '/models',
            '/health'
        ]
    })


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models),
        'active_model': 'default' if active_model else None
    })


@app.route('/models', methods=['GET'])
def list_models():
    """List loaded models."""
    return jsonify({
        'models': list(models.keys()),
        'active': 'default' if active_model else None
    })


@app.route('/models/load', methods=['POST'])
def load_model_endpoint():
    """Load a model from checkpoint."""
    data = request.json
    checkpoint_path = data.get('checkpoint')
    model_name = data.get('name', 'default')
    
    if not checkpoint_path:
        return jsonify({'error': 'checkpoint path required'}), 400
    
    try:
        load_model(checkpoint_path, model_name)
        return jsonify({
            'status': 'success',
            'model': model_name,
            'parameters': models[model_name].num_params()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/generate', methods=['POST'])
def generate():
    """Text generation endpoint."""
    if not active_model or not tokenizer:
        return jsonify({'error': 'No model loaded'}), 400
    
    data = request.json
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 50)
    temperature = data.get('temperature', 0.7)
    top_k = data.get('top_k', 0)
    top_p = data.get('top_p', 1.0)
    num_samples = data.get('num_samples', 1)
    
    try:
        # Encode prompt
        tokens = tokenizer.encode(prompt)
        
        results = []
        for _ in range(num_samples):
            keys = [[] for _ in range(active_model.n_layer)]
            values = [[] for _ in range(active_model.n_layer)]
            
            # Prime with prompt
            for i, token in enumerate(tokens[:-1]):
                _ = active_model.forward(token, i, keys, values)
            
            # Generate
            generated = active_model.generate(
                tokens[-1] if tokens else tokenizer.bos_token,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                keys=keys,
                values=values
            )
            
            text = tokenizer.decode(generated)
            results.append(text)
        
        return jsonify({
            'prompt': prompt,
            'generations': results,
            'parameters': {
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    """Chat completion endpoint (OpenAI-compatible format)."""
    if not active_model or not tokenizer:
        return jsonify({'error': 'No model loaded'}), 400
    
    data = request.json
    messages = data.get('messages', [])
    max_tokens = data.get('max_tokens', 100)
    temperature = data.get('temperature', 0.7)
    
    try:
        # Build context from messages
        context = ""
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            prefix = "User: " if role == 'user' else "Assistant: "
            context += prefix + content + "\n"
        
        context += "Assistant: "
        
        # Generate response
        tokens = tokenizer.encode(context)
        
        keys = [[] for _ in range(active_model.n_layer)]
        values = [[] for _ in range(active_model.n_layer)]
        
        for i, token in enumerate(tokens[:-1]):
            _ = active_model.forward(token, i, keys, values)
        
        generated = active_model.generate(
            tokens[-1],
            max_length=max_tokens,
            temperature=temperature
        )
        
        response = tokenizer.decode(generated).strip()
        
        # OpenAI-compatible response format
        return jsonify({
            'id': f'chatcmpl-{random.randint(100000, 999999)}',
            'object': 'chat.completion',
            'created': int(__import__('time').time()),
            'model': 'microgpt',
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': response
                },
                'finish_reason': 'stop'
            }],
            'usage': {
                'prompt_tokens': len(tokens),
                'completion_tokens': len(generated),
                'total_tokens': len(tokens) + len(generated)
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/embeddings', methods=['POST'])
def embeddings():
    """Get embeddings for text (OpenAI-compatible)."""
    if not active_model or not tokenizer:
        return jsonify({'error': 'No model loaded'}), 400
    
    data = request.json
    text = data.get('input', '')
    
    try:
        tokens = tokenizer.encode(text)
        
        keys = [[] for _ in range(active_model.n_layer)]
        values = [[] for _ in range(active_model.n_layer)]
        
        # Get final hidden state
        for i, token in enumerate(tokens):
            logits = active_model.forward(token, i, keys, values)
        
        # Use last token's representation as embedding
        embedding = [l.data for l in logits]
        
        return jsonify({
            'object': 'list',
            'data': [{
                'object': 'embedding',
                'embedding': embedding,
                'index': 0
            }],
            'model': 'microgpt',
            'usage': {
                'prompt_tokens': len(tokens),
                'total_tokens': len(tokens)
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def run_server(host='0.0.0.0', port=5000, checkpoint_path=None):
    """Run the API server."""
    if checkpoint_path:
        print(f"Loading model from {checkpoint_path}...")
        load_model(checkpoint_path)
        print("Model loaded successfully!")
    
    print(f"Starting server on {host}:{port}")
    app.run(host=host, port=port)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='microgpt API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to listen on')
    parser.add_argument('--checkpoint', help='Path to model checkpoint')
    
    args = parser.parse_args()
    run_server(args.host, args.port, args.checkpoint)
