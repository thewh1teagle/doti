"""
Hebrew diacritics encoding and decoding

Deduplicate diacritics (phonetically)
Normalize
Remove unnecessary dagesh
Sort dagesh
Convert to IDs back and forth
"""

import unicodedata
import re


# Deduplicate duplicate phonetic diacritics
NIQQUD_DEDUPLICATE = {
    "\u05b1": "\u05b5", # Hataf segol -> Tsere
    "\u05b6": "\u05b5", # Segol -> Tsere
    "\u05b2": "\u05b7", # Hataf patah -> Patah
    "\u05b3": "\u05b7", # Hataf qamats -> Patah
    "\u05b8": "\u05b7", # Qmats -> Patah
    "\u05c7": "\u05b9", # Qamats qatan -> Holam
}

NIQQUD_MAP = {
    None: 0,
    "\u05b0": 1,  # Shva
    "\u05b4": 2,  # Hirik
    "\u05b5": 3,  # Tsere
    "\u05b7": 4,  # Patah
    "\u05bb": 5,  # Qubuts
    "\u05b9": 6,  # Holam 
    "\u05ba": 7,  # Holam haser for vav
}

# UTF-8 characters names of niqqud
NIQQUD_NAME_MAP = {
    "HEBREW POINT SHEVA": "\u05b0",
    "HEBREW POINT HIRIQ": "\u05b4",
    "HEBREW POINT TSERE": "\u05b5",
    "HEBREW POINT PATAH": "\u05b7",
    "HEBREW POINT QUBUTS": "\u05bb",
    "HEBREW POINT HOLAM": "\u05b9",
    "HEBREW POINT HOLAM HASER FOR VAV": "\u05ba",
}

# Tokenization

# Shin/Sin dot
SHIN_FLAG_OFF = 0
SHIN_FLAG_RIGHT = 1
SHIN_FLAG_LEFT = 2

# Dagesh
DAGESH_FLAG_OFF = 0
DAGESH_FLAG_ON = 1

# Special tokens
PRESERVE_UKNOWN_TOKEN = -1  # Non-Hebrew but valid characters

# Hebrew letters and diacritics IDs
HEBREW_LETTERS = "אבגדהוזחטיכלמנסעפצקרשתךםןףץ"
LETTER_TO_ID = {char: idx for idx, char in enumerate(HEBREW_LETTERS)}
ID_TO_LETTER = {v: k for k, v in LETTER_TO_ID.items()}
ID_TO_NIQQUD = {v: k for k, v in NIQQUD_MAP.items()}
NIQQUD_TO_ID = {k: v for v, k in NIQQUD_MAP.items()}

def remove_niqqud(text):
    """
    Removes all diacritics from a Hebrew text.
    """
    return re.sub(r"[\u05B0-\u05C7]", "", text)

def sort_dagesh(text):
    # Alphabet followed by 1/2 symbols then dagesh. make dagesh first
    return re.sub("([\u05d0-\u05ea])([\u05b0-\u05c7]{1,2})(\u05bc)", r"\1\3\2", text)

def normalize(text):
    text = unicodedata.normalize("NFD", text)
    text = remove_unnecessary_dagesh(text)
    text = sort_dagesh(text)

    # Apply niqqud deduplication on result of dagesh removal
    normalized = []
    for char in text:
        normalized.append(NIQQUD_DEDUPLICATE.get(char, char))

    # Normalize back to NFD so diacritics stay decomposed for parsing
    return unicodedata.normalize("NFD", ''.join(normalized))

def remove_unnecessary_dagesh(text):
    # Normalize to NFD to split base and marks
    text = unicodedata.normalize("NFD", text)

    def replacer(match):
        base = match.group(0)[0]
        diacritics = match.group(0)[1:]
        if not can_dagesh(base):
            diacritics = diacritics.replace("\u05bc", "")
        return base + diacritics

    pattern = re.compile(r"[א-ת][\u05b0-\u05c7]+")
    cleaned = pattern.sub(replacer, text)
    
    return unicodedata.normalize("NFD", cleaned)

def can_dagesh(letter):
    return letter in "בכפךףו"

def can_shin_sin(letter):
    return letter in "ש"

def can_vav_holam_haser(letter):
    return letter in "ו"

def encode_hebrew_char(char, niqqud=None, dagesh_flag=DAGESH_FLAG_OFF, shin_dot_flag=SHIN_FLAG_OFF):
    """
    Encodes a single Hebrew character with diacritics into a numeric feature vector.
    """
    char_id = LETTER_TO_ID.get(char, -1)  # Handle unknowns
    niqqud_id = NIQQUD_MAP.get(niqqud, 0)
    dagesh_flag = int(dagesh_flag) if can_dagesh(char) else 0    
    return [char_id, niqqud_id, dagesh_flag, shin_dot_flag]


def encode_sentence(text):
    """
    Encodes a fully diacritized Hebrew sentence into feature vectors.
    Non-Hebrew or irrelevant characters (punctuation, digits, symbols) are replaced with UNKNOWN_TOKEN.
    Matrix look like this:
    [
        [char_id, niqqud_id, dagesh_flag, shin_flag],
        [char_id, niqqud_id, dagesh_flag, shin_flag],
        [PRESERVE_UKNOWN_TOKEN, ord(char), 0, 0],
        ...
    ]
    """
    text = normalize(text)
    
    encoded = []
    i = 0
    while i < len(text):
        char = text[i]

        # Skip punctuation, digits, Latin, etc.
        if unicodedata.category(char) != 'Lo':  # Not a Hebrew letter
            encoded.append([PRESERVE_UKNOWN_TOKEN, ord(char), 0, 0])  # Save original char's Unicode
            i += 1
            continue


        base = char
        niqqud = None
        dagesh = False
        shin_dot = None

        i += 1
        while i < len(text) and unicodedata.category(text[i]) in ('Mn', 'Sk'):
            mark = text[i]
            name = unicodedata.name(mark, "")
            
            if "DAGESH" in name:
                dagesh = True
            
            niqqud_candidate = NIQQUD_NAME_MAP.get(name, None)
            if niqqud_candidate:
                niqqud = niqqud_candidate
            elif "POINT SIN" in name:
                shin_dot = SHIN_FLAG_LEFT
            elif "POINT SHIN" in name:
                shin_dot = SHIN_FLAG_RIGHT
            elif "POINT HOLAM HASER FOR VAV" in name:
                niqqud = "\u05ba"
            i += 1
        encoded.append(encode_hebrew_char(base, niqqud, dagesh, shin_dot))

    return encoded


def decode_sentence(encoded):
    result = []
    for char_id, niqqud_id, dagesh_flag, shin_flag in encoded:
        if char_id == -1:
            result.append(chr(niqqud_id))  # SPECIAL_CHAR_TOKEN
            continue

        base_char = ID_TO_LETTER.get(char_id, "❓")
        marks = []

        if dagesh_flag:
            marks.append("\u05bc") # Dagesh
        niqqud = ID_TO_NIQQUD.get(niqqud_id, None)
        if niqqud:
            marks.append(niqqud)
        if can_shin_sin(base_char) and shin_flag:
            marks.append("\u05c1" if shin_flag == 1 else "\u05c2") # Shin/Sin

        result.append(base_char + ''.join(marks))
    return ''.join(result)


if __name__ == "__main__":
    for sentence in [
        "בְּרֵאשִׁית בָּרָא אֱלֹהִים אֵת הַשָּׁמַיִים וְאֵת הָאָרֶץ",
        # with english
        "בְּרֵאשִׁית בָּרָא אֱלֹהִים אֵת הַשָּׁמַיִים וְאֵת הָאָרֶץ Genesis 1:1",
        # with numbers
        "בְּרֵאשִׁית בָּרָא אֱלֹהִים אֵת הַשָּׁמַיִים וְאֵת הָאָרֶץ 1:1",
        # with punctuation
        "בְּרֵאשִׁית, בָּרָא אֱלֹהִים אֵת הַשָּׁמַיִים וְאֵת הָאָרֶץ",
        # with symbols
        "בְּרֵאשִׁית, בָּרָא אֱלֹהִים אֵת הַשָּמַיִים וְאֵת הָאָרֶץ 🌍",
        # with mixed languages
        "בְּרֵאשִׁית, בָּרָא אֱלֹהִים אֵת הַשָּמַיִים וְאֵת הָאָרֶץ 🌍 Genesis 1:1",
        # with unknown characters
        "בְּרֵאשִׁית, בָּרָא אֱלֹהִים אֵת הַשָּמַיִים וְאֵת הָאָרֶץ 🌍 Genesis 1:1",
        # Sofiyot
        "מְהַמֵּם כֻּלָּם"
    ]:
        sentence = normalize(sentence)
        decoded = decode_sentence(encode_sentence(sentence))
        
        assert sentence == decoded, f"Decoding failed: {sentence} != {decoded}"
        
        print(f'Original: "{sentence}"')
        print(f'Decoded: "{decoded}"')