"""
Web interface for microgpt using Flask.
Provides a simple UI for text generation.
"""

import random
from flask import Flask, render_template, request, jsonify
from pathlib import Path

from .config import Config, ModelConfig, TrainingConfig, GenerationConfig
from .model import GPT
from .data import DataLoader, CharTokenizer
from .checkpoint import CheckpointManager

app = Flask(__name__)

# Global model instance
model = None
tokenizer = None
config = None


def load_model(checkpoint_path: str = None):
    """Load model from checkpoint."""
    global model, tokenizer, config

    checkpoint_mgr = CheckpointManager()

    if checkpoint_path is None:
        checkpoint_path = checkpoint_mgr.get_latest()
        if checkpoint_path is None:
            raise ValueError("No checkpoint found")

    checkpoint = checkpoint_mgr.load_pickle(checkpoint_path)

    # Reconstruct config
    config_data = checkpoint["config"]
    config = Config(
        model=ModelConfig(**config_data.get("model", {})),
        training=TrainingConfig(**config_data.get("training", {})),
        generation=GenerationConfig(**config_data.get("generation", {})),
    )

    # Load tokenizer
    loader = DataLoader()
    loader.load_names(val_split=0)
    tokenizer = loader.tokenizer

    # Create and load model
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        block_size=config.model.block_size,
        n_layer=config.model.n_layer,
        n_embd=config.model.n_embd,
        n_head=config.model.n_head,
        dropout=0.0,
        use_gelu=config.model.use_gelu,
        use_layernorm=config.model.use_layernorm,
    )

    # Load weights
    for key, matrix in checkpoint["state_dict"].items():
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                model.state_dict[key][i][j].data = val

    model.set_training(False)

    return model, tokenizer, config


@app.route("/")
def index():
    """Render main page."""
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def generate():
    """API endpoint for text generation."""
    global model, tokenizer, config

    if model is None:
        try:
            load_model()
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    data = request.get_json()

    # Get parameters
    prompt = data.get("prompt", "")
    temperature = data.get("temperature", config.generation.temperature)
    max_length = data.get("max_length", config.generation.max_length)
    top_k = data.get("top_k", config.generation.top_k)
    top_p = data.get("top_p", config.generation.top_p)
    num_samples = data.get("num_samples", 1)
    seed = data.get("seed")

    if seed is not None:
        random.seed(seed)

    # Encode prompt
    prompt_tokens = tokenizer.encode(prompt)

    samples = []
    for _ in range(num_samples):
        # Reset cache
        keys = [[] for _ in range(model.n_layer)]
        values = [[] for _ in range(model.n_layer)]

        # Feed prompt through model
        for pos_id, token_id in enumerate(prompt_tokens):
            _ = model.forward(token_id, pos_id, keys, values)

        # Generate continuation
        start_token = prompt_tokens[-1] if prompt_tokens else tokenizer.bos_token
        start_pos = len(prompt_tokens)

        generated = []
        token_id = start_token

        for pos_id in range(start_pos, start_pos + max_length):
            logits = model.forward(token_id, pos_id, keys, values)

            # Apply temperature
            scaled_logits = [logit / temperature for logit in logits]

            # Softmax
            from model import softmax

            probs = softmax(scaled_logits)
            probs_data = [p.data for p in probs]

            # Top-k filtering
            if top_k > 0:
                sorted_probs = sorted(probs_data, reverse=True)
                threshold = sorted_probs[min(top_k, len(sorted_probs)) - 1]
                probs_data = [p if p >= threshold else 0 for p in probs_data]
                total = sum(probs_data)
                probs_data = [p / total for p in probs_data]

            # Top-p filtering
            if top_p < 1.0:
                sorted_probs = sorted(enumerate(probs_data), key=lambda x: x[1], reverse=True)
                cumsum = 0
                keep_indices = set()
                for idx, p in sorted_probs:
                    cumsum += p
                    keep_indices.add(idx)
                    if cumsum >= top_p:
                        break
                probs_data = [p if i in keep_indices else 0 for i, p in enumerate(probs_data)]
                total = sum(probs_data)
                probs_data = [p / total for p in probs_data]

            # Sample
            token_id = random.choices(range(model.vocab_size), weights=probs_data)[0]
            generated.append(token_id)

        text = tokenizer.decode(generated)
        samples.append(text)

    return jsonify(
        {
            "prompt": prompt,
            "samples": samples,
            "parameters": {
                "temperature": temperature,
                "max_length": max_length,
                "top_k": top_k,
                "top_p": top_p,
            },
        }
    )


@app.route("/api/stats", methods=["GET"])
def stats():
    """Get model statistics."""
    global model, config

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    return jsonify(
        {
            "vocab_size": model.vocab_size,
            "block_size": model.block_size,
            "n_layer": model.n_layer,
            "n_embd": model.n_embd,
            "n_head": model.n_head,
            "num_params": model.num_params(),
            "use_gelu": model.use_gelu,
            "use_layernorm": model.use_layernorm,
        }
    )


def create_templates():
    """Create HTML template for the web interface."""
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)

    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>microgpt - Text Generation</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input, textarea {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        textarea {
            min-height: 100px;
            resize: vertical;
        }
        .params {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
        }
        button {
            background: #007bff;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background: #0056b3;
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        #output {
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 4px;
            min-height: 100px;
            white-space: pre-wrap;
        }
        .sample {
            margin: 10px 0;
            padding: 10px;
            background: white;
            border-left: 3px solid #007bff;
        }
        .loading {
            text-align: center;
            color: #666;
        }
        .error {
            color: #dc3545;
        }
    </style>
</head>
<body>
    <h1>🤖 microgpt</h1>
    <div class="container">
        <div class="form-group">
            <label for="prompt">Prompt:</label>
            <textarea id="prompt" placeholder="Enter your prompt here..."></textarea>
        </div>
        
        <div class="params">
            <div class="form-group">
                <label for="temperature">Temperature:</label>
                <input type="number" id="temperature" value="0.7" min="0.1" max="2.0" step="0.1">
            </div>
            <div class="form-group">
                <label for="max_length">Max Length:</label>
                <input type="number" id="max_length" value="50" min="1" max="256">
            </div>
            <div class="form-group">
                <label for="num_samples">Samples:</label>
                <input type="number" id="num_samples" value="3" min="1" max="10">
            </div>
        </div>
        
        <div class="params">
            <div class="form-group">
                <label for="top_k">Top-k (0=off):</label>
                <input type="number" id="top_k" value="0" min="0" max="100">
            </div>
            <div class="form-group">
                <label for="top_p">Top-p (1.0=off):</label>
                <input type="number" id="top_p" value="1.0" min="0.1" max="1.0" step="0.1">
            </div>
            <div class="form-group">
                <label for="seed">Seed (optional):</label>
                <input type="number" id="seed" placeholder="Random">
            </div>
        </div>
        
        <button onclick="generate()" id="generateBtn">Generate</button>
        
        <div id="output"></div>
    </div>

    <script>
        async function generate() {
            const btn = document.getElementById('generateBtn');
            const output = document.getElementById('output');
            
            btn.disabled = true;
            output.innerHTML = '<div class="loading">Generating...</div>';
            
            const params = {
                prompt: document.getElementById('prompt').value,
                temperature: parseFloat(document.getElementById('temperature').value),
                max_length: parseInt(document.getElementById('max_length').value),
                num_samples: parseInt(document.getElementById('num_samples').value),
                top_k: parseInt(document.getElementById('top_k').value),
                top_p: parseFloat(document.getElementById('top_p').value),
                seed: document.getElementById('seed').value ? 
                      parseInt(document.getElementById('seed').value) : null
            };
            
            try {
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(params)
                });
                
                const data = await response.json();
                
                if (data.error) {
                    output.innerHTML = `<div class="error">Error: ${data.error}</div>`;
                } else {
                    let html = `<strong>Prompt:</strong> ${data.prompt || '(empty)'}<br><br>`;
                    html += '<strong>Generated samples:</strong>';
                    data.samples.forEach((sample, i) => {
                        html += `<div class="sample">${i+1}. ${sample}</div>`;
                    });
                    output.innerHTML = html;
                }
            } catch (err) {
                output.innerHTML = `<div class="error">Error: ${err.message}</div>`;
            } finally {
                btn.disabled = false;
            }
        }
    </script>
</body>
</html>"""

    template_path = templates_dir / "index.html"
    with open(template_path, "w") as f:
        f.write(html_content)

    print(f"Template created at {template_path}")


def main():
    """Run the web application."""
    import sys

    # Create templates if they don't exist
    if not Path("templates/index.html").exists():
        create_templates()

    # Try to load model
    try:
        load_model()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("Please train a model first using: python cli.py train")

    # Run Flask
    port = 5000
    if len(sys.argv) > 1:
        port = int(sys.argv[1])

    print(f"\nStarting server on http://localhost:{port}")
    print("Press Ctrl+C to stop")
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
