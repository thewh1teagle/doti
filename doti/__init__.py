from doti.predict import predict
from doti.model import build_model

import torch

class Doti:
    def __init__(self, model_path: str):
        self.model = build_model()
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def compute(self, text: str) -> str:
        return predict(self.model, text)
