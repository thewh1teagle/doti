import numpy as np
import onnxruntime as ort
from doti.model.tokenizer import encode_sentence, decode_sentence, remove_niqqud

def predict(session: ort.InferenceSession, text):
    # Preprocess input
    stripped = encode_sentence(remove_niqqud(text))
    X = [tok[0] for tok in stripped if tok[0] >= 0]
    input_array = np.array([X], dtype=np.int64)  # ONNX expects int64

    # Run inference
    outputs = session.run(
        None,  # Output all names
        {"input": input_array}
    )

    # Get predictions
    niqqud_pred = np.argmax(outputs[0], axis=-1)[0].tolist()
    dagesh_pred = np.argmax(outputs[1], axis=-1)[0].tolist()
    shin_pred = np.argmax(outputs[2], axis=-1)[0].tolist()

    # Reconstruct sentence
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
