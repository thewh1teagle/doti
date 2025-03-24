"""
pip install '.[onnx]'
"""
import onnxruntime
from doti import DotiOnnx

session = onnxruntime.InferenceSession('model.onnx')
doti = DotiOnnx(session)
sentence = "שלום עולם"
result = doti.compute(sentence)
print(result)