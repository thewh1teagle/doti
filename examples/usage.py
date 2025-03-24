from doti import Doti
from doti.dataset import load_dataset
from doti.tokenizer import remove_niqqud

if __name__ == "__main__":
    sentences = load_dataset()
    model = Doti("ckpt/checkpoint_20764.pth")
    for with_niqqud in sentences:
        without_niqqud = remove_niqqud(with_niqqud)
        print(f'Input sentence: {with_niqqud}')
        print(f'Predicted sentence: {model.compute(without_niqqud)}')