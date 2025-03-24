"""
pip install '.[torch]'
"""

from doti import Doti
from doti.train.dataset import load_dataset
from doti.model.tokenizer import remove_niqqud

if __name__ == "__main__":
    sentences = load_dataset()
    sentences = ["שלום עולם"]
    model = Doti("ckpt/checkpoint_1000.pth")
    for with_niqqud in sentences:
        without_niqqud = remove_niqqud(with_niqqud)
        print(f'Input sentence: {with_niqqud}')
        print(f'Predicted sentence: {model.compute(without_niqqud)}')