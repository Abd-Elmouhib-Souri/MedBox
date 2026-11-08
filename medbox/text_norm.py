"""Text normalization shared by the drug list builder, the matcher and the evaluation.

One function, used everywhere: the list, the OCR output and the ground truth
are always compared in the same normalized form.
"""

import re
import unicodedata

# A "dosage token" is a number, optionally with a unit, optionally a ratio: 50, 500MG, 1G, 2.5, 100/25, 5%, 1000UI
_DOSAGE_TOKEN = re.compile(
    r"^\d+([.,]\d+)?(MG|G|ML|UI|MCG|UG|%)?(/\d+([.,]\d+)?(MG|G|ML|UI|MCG|UG|%)?)?$"
)
_UNITS = {"MG", "G", "ML", "UI", "MCG", "UG", "%"}               # a unit written as its own token


def normalize(text: str) -> str:
    """Uppercase, remove accents, keep letters/digits/basic signs, collapse spaces."""
    text = unicodedata.normalize("NFKD", str(text))                 # "é" -> "e" + accent mark
    text = "".join(c for c in text if not unicodedata.combining(c))  # drop the accent marks
    text = text.upper()
    text = re.sub(r"[^A-Z0-9%/.,+\- ]", " ", text)                  # other symbols become spaces
    return re.sub(r"\s+", " ", text).strip()                         # many spaces -> one space


def base_name(text: str) -> str:
    """Normalized trade name without trailing dosage tokens.

    "CARBOLESS 50" -> "CARBOLESS", "AUGMENTIN 1G" -> "AUGMENTIN".
    Only tokens at the END that are pure dosages are removed: "VITAMINE B12" stays unchanged,
    because "B12" starts with a letter.
    If removing would leave nothing, the normalized text is returned unchanged.
    """
    tokens = normalize(text).split(" ")
    while len(tokens) > 1:                                          # strip from the end only
        if _DOSAGE_TOKEN.match(tokens[-1]):                         # "50", "500MG", "20/12,5"
            tokens.pop()
        elif (len(tokens) > 2 and tokens[-1] in _UNITS
              and _DOSAGE_TOKEN.match(tokens[-2])):                 # "2.5 MG": number and unit split by a space
            del tokens[-2:]
        else:
            break
    return " ".join(tokens)
