#!/usr/bin/env python3
"""Export McCarthyGPT to ONNX for browser inference."""

import sys
import os
import json
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models", "v0"))
from model import McCarthyGPT, Config


class McCarthyGPTForExport(nn.Module):
    """Wrapper that returns only logits (no loss tuple) for ONNX export."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, idx):
        logits, _ = self.model(idx)
        return logits


def main():
    checkpoint_path = "checkpoints/final_modal.pt"
    output_dir = "site"

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    meta = ckpt["meta"]

    # Checkpoint was trained with ConfigV1 dims but only saved vocab_size + block_size.
    # Override to match actual weight shapes.
    config.n_embd = 256
    config.n_head = 8
    config.n_layer = 6
    config.dropout = 0.1
    config.bias = False

    print(f"Config: vocab_size={config.vocab_size}, block_size={config.block_size}, "
          f"n_embd={config.n_embd}, n_head={config.n_head}, n_layer={config.n_layer}")

    # Build model matching the checkpoint
    model = McCarthyGPT(config)
    model.load_state_dict(ckpt["model"])
    model.eval()

    wrapper = McCarthyGPTForExport(model)
    wrapper.eval()

    # Export ONNX
    dummy_input = torch.randint(0, config.vocab_size, (1, 16), dtype=torch.long)
    onnx_path = os.path.join(output_dir, "mccarthy.onnx")

    print(f"Exporting to {onnx_path}...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        onnx_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {1: "seq_len"},
            "logits": {1: "seq_len"},
        },
        opset_version=18,
    )

    # Merge external data into single file for browser compatibility
    import onnx
    from onnx.external_data_helper import convert_model_to_external_data

    data_path = onnx_path + ".data"
    if os.path.exists(data_path):
        print("Merging external data into single ONNX file...")
        model_proto = onnx.load(onnx_path, load_external_data=True)
        # Clear external data references and embed weights
        for tensor in model_proto.graph.initializer:
            tensor.ClearField("data_location")
            if tensor.HasField("raw_data") is False:
                pass  # already embedded
        onnx.save(model_proto, onnx_path)
        os.remove(data_path)

    onnx_size = os.path.getsize(onnx_path)
    print(f"ONNX model: {onnx_size / 1e6:.1f} MB")

    # Export vocabulary as JSON
    vocab = {
        "stoi": meta["stoi"],
        "itos": {str(k): v for k, v in meta["itos"].items()},
        "vocab_size": meta["vocab_size"],
        "block_size": config.block_size,
    }

    vocab_path = os.path.join(output_dir, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(vocab, f)

    print(f"Vocab: {vocab_path} ({len(meta['stoi'])} chars)")

    # Verify ONNX model
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_path)
        test_input = torch.randint(0, config.vocab_size, (1, 10)).numpy()
        result = sess.run(None, {"input_ids": test_input})
        print(f"Verification: input {test_input.shape} -> output {result[0].shape}")
        print("ONNX export verified.")
    except ImportError:
        print("onnxruntime not installed, skipping verification.")
        print("Install with: pip install onnxruntime")

    print("Done.")


if __name__ == "__main__":
    main()
