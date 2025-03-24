import torch.nn as nn
from doti.tokenizer import LETTER_TO_ID, NIQQUD_TO_ID

config = {
    "embedding_dim": 32,
    "hidden_dim": 64,
    "vocab_size": len(LETTER_TO_ID),
    "niqqud_classes": len(NIQQUD_TO_ID),
    "dagesh_classes": 2, # OFF, ON
    "shin_classes": 3 # OFF, SHIN, SIN
}

# Hyperparameters
EMBEDDING_DIM = 32
HIDDEN_DIM = 64

class DotiModel(nn.Module):
    def __init__(self, config):
        super(DotiModel, self).__init__()
        self.embedding = nn.Embedding(config["vocab_size"], config["embedding_dim"])
        self.lstm = nn.LSTM(config["embedding_dim"], config["hidden_dim"], batch_first=True)

        # 3 output heads
        self.fc_niqqud = nn.Linear(config["hidden_dim"], config["niqqud_classes"])
        self.fc_dagesh = nn.Linear(config["hidden_dim"], config["dagesh_classes"])
        self.fc_shin = nn.Linear(config["hidden_dim"], config["shin_classes"])

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)

        niqqud_out = self.fc_niqqud(out)
        dagesh_out = self.fc_dagesh(out)
        shin_out = self.fc_shin(out)

        return niqqud_out, dagesh_out, shin_out

def build_model():
    return DotiModel(config)