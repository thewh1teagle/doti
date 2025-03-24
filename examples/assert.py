"""
pip install '.[torch]'
"""

from doti import Doti
from doti.model.tokenizer import remove_niqqud, normalize
from doti.train.dataset import load_dataset

if __name__ == "__main__":
    sentences = load_dataset()
    model = Doti("ckpt/checkpoint_20764.pth")
    for with_niqqud in sentences:
        with_niqqud = normalize(with_niqqud)
        without_niqqud = remove_niqqud(with_niqqud)
        with_niqqud_prediction = model.compute(without_niqqud)
        assert with_niqqud_prediction == with_niqqud, f"{with_niqqud_prediction} != {with_niqqud}"
        print(f"In: {without_niqqud}")
        print(f"Out: {with_niqqud_prediction}")