import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


class Preprocessor:
    def __init__(self):
        self._stop = ENGLISH_STOP_WORDS

    def preprocess(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def tokenize(self, text: str) -> list[str]:
        return [w for w in self.preprocess(text).split() if w not in self._stop]
