"""
uv pip install '.[onnx]'
"""

import torch
from model import build_model, config

# Build the model
model = build_model()
model.eval()  # Set to eval mode for export

# Create dummy input: (batch_size, sequence_length)
batch_size = 1
seq_len = 10  # Change as needed
dummy_input = torch.randint(0, config["vocab_size"], (batch_size, seq_len), dtype=torch.long)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",  # Output file name
    input_names=["input"],
    output_names=["niqqud_out", "dagesh_out", "shin_out"],
    dynamic_axes={
        "input": {0: "batch_size", 1: "seq_len"},
        "niqqud_out": {0: "batch_size", 1: "seq_len"},
        "dagesh_out": {0: "batch_size", 1: "seq_len"},
        "shin_out": {0: "batch_size", 1: "seq_len"},
    },
    opset_version=11  # ONNX opset version, 11+ is usually fine
)

print("Model exported to doti_model.onnx")
