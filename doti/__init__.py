from doti.model.predict import predict
from doti.model import build_model
import torch

class Doti:
    def __init__(self, model_path: str):
        self.model = build_model()
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def compute(self, text: str) -> str:
        return predict(self.model, text)

class DotiOnnx:
    def __init__(self, session):
        self.session = session
    
    def compute(self, text):
        from doti.onnx.predict import predict
        return predict(self.session, text)