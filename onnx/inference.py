import onnxruntime as rt
from transformers import AutoTokenizer
import numpy as np

import torch

MODEL_PATH = ""

sess = rt.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


def infer(sentence):
    model_inputs = tokenizer(
        sentence, padding=True, truncation=True, max_length=128, return_tensors="pt"
    )

    inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}
    sequence = sess.run(None, inputs_onnx)
    return sequence


def _mean_pooling(model_output, attention_mask):
    token_embeddings = torch.from_numpy(model_output[0])
    input_mask_expanded = torch.from_numpy(attention_mask).unsqueeze(-1).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask, input_mask_expanded, sum_mask


def sentence_embedding(sentence: str, normalize: bool = False) -> np.ndarray:
    model_inputs = tokenizer(
        sentence, padding=True, truncation=True, max_length=128, return_tensors="pt"
    )

    inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}
    sequence = sess.run(None, inputs_onnx)
    embeddings = _mean_pooling(sequence, inputs_onnx["attention_mask"])[0][0]

    if normalize:
        embeddings = embeddings / np.linalg.norm(embeddings)

    return embeddings.numpy()
