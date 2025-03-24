import torch
import torch.nn as nn
import torch.optim as optim
from model import build_model
from tokenizer import encode_sentence, remove_niqqud
from dataset import load_dataset, train_test_split

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
EPOCHS = 10000  # Use a smaller value (e.g., 100) for testing purposes
LEARNING_RATE = 0.01


def prepare_data(sentences):
    X_batch, y_niqqud, y_dagesh, y_shin = [], [], [], []

    for ex in sentences:
        encoded = encode_sentence(ex)
        stripped = encode_sentence(remove_niqqud(ex))

        X, niqqud, dagesh, shin = [], [], [], []

        for stripped_tok, full_tok in zip(stripped, encoded):
            char_id, _, _, _ = stripped_tok
            _, niqqud_id, dagesh_flag, shin_flag = full_tok

            if char_id >= 0:
                X.append(char_id)
                niqqud.append(niqqud_id if niqqud_id is not None else 0)
                dagesh.append(dagesh_flag if dagesh_flag is not None else 0)
                shin.append(shin_flag if shin_flag is not None else 0)

        X_batch.append(X)
        y_niqqud.append(niqqud)
        y_dagesh.append(dagesh)
        y_shin.append(shin)

    max_len = max(len(seq) for seq in X_batch)
    for i in range(len(X_batch)):
        pad_len = max_len - len(X_batch[i])
        X_batch[i] += [0] * pad_len
        y_niqqud[i] += [0] * pad_len
        y_dagesh[i] += [0] * pad_len
        y_shin[i] += [0] * pad_len

    return (
        torch.tensor(X_batch, dtype=torch.long).to(device),
        torch.tensor(y_niqqud, dtype=torch.long).to(device),
        torch.tensor(y_dagesh, dtype=torch.long).to(device),
        torch.tensor(y_shin, dtype=torch.long).to(device),
    )


def train(sentences: list):
    X, y_niqqud, y_dagesh, y_shin = prepare_data(sentences)

    model = build_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        niqqud_out, dagesh_out, shin_out = model(X)

        loss_niqqud = criterion(niqqud_out.view(-1, niqqud_out.shape[-1]), y_niqqud.view(-1))
        loss_dagesh = criterion(dagesh_out.view(-1, dagesh_out.shape[-1]), y_dagesh.view(-1))
        loss_shin = criterion(shin_out.view(-1, shin_out.shape[-1]), y_shin.view(-1))

        loss = loss_niqqud + loss_dagesh + loss_shin
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

    return model


def evaluate(model, sentences):
    model.eval()
    X, y_niqqud, y_dagesh, y_shin = prepare_data(sentences)

    with torch.no_grad():
        niqqud_out, dagesh_out, shin_out = model(X)

        pred_niqqud = niqqud_out.argmax(dim=-1)
        pred_dagesh = dagesh_out.argmax(dim=-1)
        pred_shin = shin_out.argmax(dim=-1)

        accuracy_niqqud = (pred_niqqud == y_niqqud).float().mean().item()
        accuracy_dagesh = (pred_dagesh == y_dagesh).float().mean().item()
        accuracy_shin = (pred_shin == y_shin).float().mean().item()

        print("Accuracy on unseen data:")
        print(f"  Niqqud:  {accuracy_niqqud:.2%}")
        print(f"  Dagesh:  {accuracy_dagesh:.2%}")
        print(f"  Shin:    {accuracy_shin:.2%}")


if __name__ == "__main__":
    sentences = load_dataset()
    train_data, test_data = train_test_split(sentences)
    model = train(train_data)
    evaluate(model, test_data)
    torch.save(model.state_dict(), "model.pth")
    print("Model saved as model.pth")