# Doti

Add diacritics to Hebrew text


```python
"""
pip install '.[torch]'
"""

from doti import Doti
from doti.dataset import load_dataset
from doti.tokenizer import remove_niqqud

if __name__ == "__main__":
    sentences = load_dataset()
    model = Doti("model.pth")
    niqqud = model.compute("שלום עולם")
    print(niqqud)
```