import string
import re
import unicodedata
from unidecode import unidecode
import random

def canonicalize(text: str) -> str:
    if text is None:
        return ""

    text = text.lower()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def normalize_text(text):
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower().strip()
    return text

def preprocess_sentence(sentence: str):
    sentence = sentence.lower()
    sentence = unicodedata.normalize('NFC', sentence)
    sentence = re.sub(r"[“”]", "\"", sentence)
    sentence = re.sub(r"!", " ! ", sentence)
    sentence = re.sub(r"\?", " ? ", sentence)
    sentence = re.sub(r":", " : ", sentence)
    sentence = re.sub(r";", " ; ", sentence)
    sentence = re.sub(r",", " , ", sentence)
    sentence = re.sub(r"\"", " \" ", sentence)
    sentence = re.sub(r"'", " ' ", sentence)
    sentence = re.sub(r"\(", " ( ", sentence)
    sentence = re.sub(r"\[", " [ ", sentence)
    sentence = re.sub(r"\)", " ) ", sentence)
    sentence = re.sub(r"\]", " ] ", sentence)
    sentence = re.sub(r"/", " / ", sentence)
    sentence = re.sub(r"\.", " . ", sentence)
    sentence = re.sub(r"-", " - ", sentence)
    sentence = re.sub(r"\$", " $ ", sentence)
    sentence = re.sub(r"\&", " & ", sentence)
    sentence = re.sub(r"\*", " * ", sentence)
    return sentence

def remove_vietnamese_accents(sentence, ratio=0.5):
    output = ''
    for char in sentence:
        if char in 'áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộớờởỡợúùủũụưứừửữựýỳỷỹỵ':
            if random.random() < ratio:
                output += unidecode(char)
            else:
                output += char
        else:
            output += char
    return output

def remove_repeated_substrings(text: str) -> str:
    pattern = re.compile(r'(.+?)\1+')

    prev = None
    while prev != text:
        prev = text
        text = pattern.sub(r'\1', text)

    return text