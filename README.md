# Doti

Add diacritics to Hebrew text


```python
"""
pip install '.[torch]'
"""

from doti import Doti

if __name__ == "__main__":
    model = Doti("model.pth")
    niqqud = model.compute("שלום עולם")
    print(niqqud)
```
