"""
pip install '.[onnx]'
"""
import onnxruntime
from doti.onnx.predict import predict_onnx

session = onnxruntime.InferenceSession('model.onnx')
sentence = "שלום עולם"
result = predict_onnx(session, sentence)
print(result)