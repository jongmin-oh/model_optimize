import transformers
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import transformers.convert_graph_to_onnx as onnx_convert

MODEL_PATH = ""

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

pipeline = transformers.pipeline(
    "text-classification", model=model, tokenizer=tokenizer
)

model = model.to("cpu")

onnx_convert.convert_pytorch(
    pipeline,
    opset=17,
    output=Path("model.onnx"),
    use_external_format=False,
)
