#!/usr/bin/env python3
"""
McCarthyGPT Web Interface

A Gradio app for generating text in the style of Cormac McCarthy.
Trained on Blood Meridian using a character-level transformer.

Usage:
    pip install gradio
    python app.py

Then open http://localhost:7860 in your browser.
For public sharing: python app.py --share
"""

import argparse
import torch
from models.v0.model import McCarthyGPT

# Global model (loaded once)
MODEL = None
META = None  
DEVICE = None


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
    return MODEL, META, DEVICE


def generate_text(prompt: str, max_tokens: int = 500, temperature: float = 0.8) -> str:
    """Generate McCarthy-style text from a prompt."""
    if MODEL is None:
        return "Error: Model not loaded"
    
    if not prompt.strip():
        prompt = "The"
    
    stoi, itos = META['stoi'], META['itos']
    
    # Encode prompt (skip unknown chars)
    tokens = [stoi[c] for c in prompt if c in stoi]
    if not tokens:
        tokens = [0]
    
    x = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
    
    with torch.no_grad():
        out = MODEL.generate(x, max_new_tokens=max_tokens, temperature=temperature)
    
    return ''.join([itos[i] for i in out[0].tolist()])


def create_app():
    """Create the Gradio interface."""
    import gradio as gr
    
    # Example prompts
    examples = [
        ["The judge"],
        ["They rode"],
        ["The desert"],
        ["He stood in the doorway"],
        ["The man"],
        ["At dawn"],
    ]
    
    with gr.Blocks(title="McCarthyGPT", theme=gr.themes.Base()) as app:
        gr.Markdown("""
        # 🏜️ McCarthyGPT
        
        A character-level language model trained on Cormac McCarthy's prose style.
        Enter a prompt and watch it generate text with that distinctive sparse, 
        biblical, blood-soaked Western voice.
        
        **Model:** 4.81M parameters | **Training:** 5000 steps on T4 GPU | **Architecture:** GPT (6 layers, 6 heads)
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="The judge...",
                    lines=2,
                )
                
                with gr.Row():
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.8,
                        step=0.1,
                        label="Temperature",
                        info="Lower = more focused, Higher = more creative"
                    )
                    max_tokens = gr.Slider(
                        minimum=100,
                        maximum=1000,
                        value=500,
                        step=50,
                        label="Max Length",
                        info="Characters to generate"
                    )
                
                generate_btn = gr.Button("Generate", variant="primary")
            
            with gr.Column(scale=3):
                output = gr.Textbox(
                    label="Generated Text",
                    lines=15,
                    show_copy_button=True,
                )
        
        gr.Examples(
            examples=examples,
            inputs=[prompt_input],
            label="Try these prompts:"
        )
        
        gr.Markdown("""
        ---
        *Built by [@theapexwu](https://github.com/theapexwu) | 
        Part of [BloodMeridianNLP](https://github.com/theapexwu/BloodMeridianNLP)*
        """)
        
        # Event handlers
        generate_btn.click(
            fn=generate_text,
            inputs=[prompt_input, max_tokens, temperature],
            outputs=output,
        )
        
        prompt_input.submit(
            fn=generate_text,
            inputs=[prompt_input, max_tokens, temperature],
            outputs=output,
        )
    
    return app


def main():
    parser = argparse.ArgumentParser(description="McCarthyGPT Web App")
    parser.add_argument('--checkpoint', default='checkpoints/final_modal.pt', help='Model checkpoint')
    parser.add_argument('--share', action='store_true', help='Create public Gradio link')
    parser.add_argument('--port', type=int, default=7860, help='Port to run on')
    args = parser.parse_args()
    
    # Load model
    load_model(args.checkpoint)
    
    # Create and launch app
    app = create_app()
    app.launch(
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0",
    )


if __name__ == '__main__':
    main()
