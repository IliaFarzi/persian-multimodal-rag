from hazm import Normalizer, word_tokenize

class PersianPreprocessor:
    def __init__(self):
        self.normalizer = Normalizer()

    def normalize(self, text):
        return self.normalizer.normalize(text)

    def tokenize(self, text):
        return word_tokenize(text)