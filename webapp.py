#!/usr/bin/env python3
"""
McCarthyGPT Simple Web Interface

A lightweight Flask app for generating text in McCarthy's style.

Usage:
    pip install flask
    python webapp.py

Then open http://localhost:5000
"""

from flask import Flask, render_template_string, request, jsonify
import torch
from model import McCarthyGPT

app = Flask(__name__)

# Global model
MODEL = None
META = None
DEVICE = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🏜️ McCarthyGPT</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: Georgia, 'Times New Roman', serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 40px 20px;
            background: #1a1a1a;
            color: #e0e0e0;
        }
        h1 { 
            color: #c9a959;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #888;
            margin-bottom: 40px;
            font-style: italic;
        }
        .input-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            color: #c9a959;
        }
        textarea {
            width: 100%;
            padding: 15px;
            font-size: 16px;
            font-family: Georgia, serif;
            background: #2a2a2a;
            border: 1px solid #444;
            color: #e0e0e0;
            border-radius: 4px;
        }
        textarea:focus {
            outline: none;
            border-color: #c9a959;
        }
        .controls {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }
        .control {
            flex: 1;
        }
        input[type="range"] {
            width: 100%;
        }
        .value {
            text-align: right;
            color: #888;
            font-size: 14px;
        }
        button {
            width: 100%;
            padding: 15px 30px;
            font-size: 18px;
            font-family: Georgia, serif;
            background: #8b4513;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            transition: background 0.2s;
        }
        button:hover {
            background: #a0522d;
        }
        button:disabled {
            background: #444;
            cursor: not-allowed;
        }
        #output {
            margin-top: 30px;
            padding: 25px;
            background: #2a2a2a;
            border-left: 4px solid #c9a959;
            min-height: 200px;
            line-height: 1.8;
            white-space: pre-wrap;
            font-size: 16px;
        }
        .loading {
            color: #888;
            font-style: italic;
        }
        .examples {
            margin-top: 20px;
            text-align: center;
        }
        .examples span {
            display: inline-block;
            padding: 8px 15px;
            margin: 5px;
            background: #333;
            border-radius: 20px;
            cursor: pointer;
            font-size: 14px;
        }
        .examples span:hover {
            background: #444;
        }
        footer {
            margin-top: 40px;
            text-align: center;
            color: #666;
            font-size: 14px;
        }
        footer a { color: #c9a959; }
    </style>
</head>
<body>
    <h1>🏜️ McCarthyGPT</h1>
    <p class="subtitle">A character-level language model trained on Cormac McCarthy's prose</p>
    
    <div class="input-group">
        <label>Prompt</label>
        <textarea id="prompt" rows="2" placeholder="The judge...">The judge</textarea>
    </div>
    
    <div class="controls">
        <div class="control">
            <label>Temperature</label>
            <input type="range" id="temp" min="0.1" max="2.0" step="0.1" value="0.8">
            <div class="value" id="temp-value">0.8</div>
        </div>
        <div class="control">
            <label>Length</label>
            <input type="range" id="length" min="100" max="1000" step="50" value="500">
            <div class="value" id="length-value">500</div>
        </div>
    </div>
    
    <button id="generate" onclick="generate()">Generate</button>
    
    <div class="examples">
        <span onclick="setPrompt('The judge')">The judge</span>
        <span onclick="setPrompt('They rode')">They rode</span>
        <span onclick="setPrompt('The desert')">The desert</span>
        <span onclick="setPrompt('At dawn')">At dawn</span>
        <span onclick="setPrompt('The man')">The man</span>
    </div>
    
    <div id="output"></div>
    
    <footer>
        4.81M parameters · Trained on T4 GPU · 
        <a href="https://github.com/theapexwu/BloodMeridianNLP">GitHub</a>
    </footer>
    
    <script>
        document.getElementById('temp').oninput = function() {
            document.getElementById('temp-value').textContent = this.value;
        };
        document.getElementById('length').oninput = function() {
            document.getElementById('length-value').textContent = this.value;
        };
        
        function setPrompt(text) {
            document.getElementById('prompt').value = text;
        }
        
        async function generate() {
            const prompt = document.getElementById('prompt').value;
            const temp = document.getElementById('temp').value;
            const length = document.getElementById('length').value;
            const btn = document.getElementById('generate');
            const output = document.getElementById('output');
            
            btn.disabled = true;
            btn.textContent = 'Generating...';
            output.innerHTML = '<span class="loading">The desert wind rises...</span>';
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({prompt, temperature: parseFloat(temp), max_tokens: parseInt(length)})
                });
                const data = await response.json();
                output.textContent = data.text;
            } catch (e) {
                output.textContent = 'Error: ' + e.message;
            }
            
            btn.disabled = false;
            btn.textContent = 'Generate';
        }
        
        // Allow Ctrl+Enter to generate
        document.getElementById('prompt').onkeydown = function(e) {
            if (e.ctrlKey && e.key === 'Enter') generate();
        };
    </script>
</body>
</html>
"""


def load_model(checkpoint_path="checkpoints/final_modal.pt"):
    """Load the trained model."""
    global MODEL, META, DEVICE
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Loading McCarthyGPT on {DEVICE}...")
    
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    config = ckpt['config']
    META = ckpt['meta']
    
    MODEL = McCarthyGPT(config).to(DEVICE)
    MODEL.load_state_dict(ckpt['model'])
    MODEL.eval()
    
    print(f"Model loaded: {sum(p.numel() for p in MODEL.parameters()):,} parameters")


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/generate', methods=['POST'])
def generate():
    import re
    
    data = request.json
    prompt = data.get('prompt', 'The')
    temperature = data.get('temperature', 0.8)
    max_tokens = data.get('max_tokens', 500)
    
    if not prompt.strip():
        prompt = "The"
    
    stoi, itos = META['stoi'], META['itos']
    
    # Encode prompt
    tokens = [stoi[c] for c in prompt if c in stoi]
    if not tokens:
        tokens = [0]
    
    x = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
    
    with torch.no_grad():
        out = MODEL.generate(x, max_new_tokens=max_tokens, temperature=temperature)
    
    text = ''.join([itos[i] for i in out[0].tolist()])
    
    # Clean up spacing artifacts from char-level generation
    text = re.sub(r'\n+', ' ', text)      # newlines -> spaces
    text = re.sub(r'  +', ' ', text)      # multiple spaces -> single
    text = re.sub(r' ([.,;:!?])', r'\1', text)  # no space before punctuation
    
    return jsonify({'text': text})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/final_modal.pt')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    
    load_model(args.checkpoint)
    print(f"\n🏜️ McCarthyGPT running at http://localhost:{args.port}\n")
    app.run(host='0.0.0.0', port=args.port, debug=False)
