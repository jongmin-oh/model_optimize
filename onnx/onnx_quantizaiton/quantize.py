from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    "encoder.onnx", "encoder.onnx_uint8.onnx", weight_type=QuantType.QUInt8
)
