import torch.nn as nn
from doti.tokenizer import LETTER_TO_ID, NIQQUD_TO_ID

# Hyperparameters
EMBEDDING_DIM = 32
HIDDEN_DIM = 64

class DotiModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, niqqud_classes, dagesh_classes=2, shin_classes=3):
        super(DotiModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.lstm = nn.LSTM(EMBEDDING_DIM, hidden_dim, batch_first=True)

        # 3 output heads
        self.fc_niqqud = nn.Linear(hidden_dim, niqqud_classes)
        self.fc_dagesh = nn.Linear(hidden_dim, dagesh_classes)
        self.fc_shin = nn.Linear(hidden_dim, shin_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)

        niqqud_out = self.fc_niqqud(out)
        dagesh_out = self.fc_dagesh(out)
        shin_out = self.fc_shin(out)

        return niqqud_out, dagesh_out, shin_out

def build_model():
    return DotiModel(
        vocab_size=len(LETTER_TO_ID),
        hidden_dim=HIDDEN_DIM,
        niqqud_classes=len(NIQQUD_TO_ID)
    )