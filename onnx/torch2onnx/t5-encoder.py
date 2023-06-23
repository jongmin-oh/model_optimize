from pathlib import Path

from transformers import T5EncoderModel, AutoTokenizer
from transformers import pipeline
import transformers.convert_graph_to_onnx as onnx_convert

MODEL_PATH = "j5ng/et5-base"

T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
auto_model = T5EncoderModel.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

encoder = pipeline(
    "feature-extraction", model=auto_model, tokenizer=tokenizer, return_tensors=True
)

onnx_convert.convert_pytorch(
    encoder, opset=17, output=Path("encoder.onnx"), use_external_format=False
)
