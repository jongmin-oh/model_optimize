from pathlib import Path
from transformers.convert_graph_to_onnx import convert

MODEL_PATH = ""

convert(
    framework="pt",
    model=MODEL_PATH,
    output=Path("model.onnx"),
    opset=17,
)
