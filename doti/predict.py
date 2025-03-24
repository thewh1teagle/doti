from .tokenizer import encode_sentence, decode_sentence, remove_niqqud, ID_TO_LETTER, LETTER_TO_ID, ID_TO_NIQQUD, NIQQUD_TO_ID
import torch

# Inference
def predict(model, text):
    stripped = encode_sentence(remove_niqqud(text))
    X = [tok[0] for tok in stripped if tok[0] >= 0]
    input_tensor = torch.tensor([X], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        niqqud_out, dagesh_out, shin_out = model(input_tensor)
        niqqud_pred = torch.argmax(niqqud_out, dim=-1)[0].tolist()
        dagesh_pred = torch.argmax(dagesh_out, dim=-1)[0].tolist()
        shin_pred = torch.argmax(shin_out, dim=-1)[0].tolist()

    reconstructed = []
    idx = 0
    for tok in stripped:
        char_id, _, _, _ = tok
        if char_id >= 0:
            reconstructed.append([
                char_id,
                niqqud_pred[idx],
                dagesh_pred[idx],
                shin_pred[idx]
            ])
            idx += 1
        else:
            reconstructed.append(tok)

    return decode_sentence(reconstructed)
