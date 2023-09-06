import numpy as np

class WordVectorizer:
    def __init__(self, corpus=None):
        self.char2index = {char: idx for idx, char in enumerate("abcdefghijklmnopqrstuvwxyz")}

        # Continuous Encoding Values
        self.continuous_encoding = {char: (idx + 1) * 0.1 for idx, char in enumerate("abcdefghijklmnopqrstuvwxyz")}
        self.continuous_encoding[" "] = 0.0

        self.vector_size = len(self.char2index)

        if corpus:
            self.corpus = corpus
            self.idf = self.calculate_idf()
        else:
            self.corpus = None
            self.idf = None

    def one_hot_vectorize(self, word):
        word_vector = np.zeros((len(word), self.vector_size), dtype=float)
        for i, char in enumerate(word.lower()):
            if char in self.char2index:
                word_vector[i, self.char2index[char]] = 1.0
        return word_vector

    def continuous_vectorize(self, word):
        return np.array([self.continuous_encoding[char] for char in word.lower() if char in self.continuous_encoding],
                        dtype=float)

    def binary_vectorize(self, word):
        # Vectorize using the binary representation of ASCII values of characters
        return np.array([list(map(int, format(ord(char), '08b'))) for char in word.lower()], dtype=float)

    def frequency_vectorize(self, word):
        # Vectorize by the frequency of each letter in the word
        freq = [word.count(char) for char in self.char2index]
        return np.array(freq, dtype=float) / len(word)  # Normalize by word length

    def ngram_vectorize(self, word, n=2):
        # Vectorize by creating n-grams
        ngrams = [word[i:i + n] for i in range(len(word) - n + 1)]
        ngram_vector = []
        for ng in ngrams:
            vec = []
            for char in ng:
                if char in self.char2index:
                    vec.append(self.char2index[char])
            ngram_vector.append(vec)
        return np.array(ngram_vector, dtype=float)

    def calculate_idf(self):
        num_words = len(self.corpus)
        idf_dict = {}
        for char in self.char2index:
            count_words_with_char = sum(1 for word in self.corpus if char in word)
            idf_dict[char] = np.log(num_words / (1 + count_words_with_char))  # 1 is added to avoid division by zero
        return idf_dict

    def tfidf_vectorize(self, word):
        if not self.idf:
            raise ValueError("TF-IDF requires a corpus for initialization.")
        tfidf_vector = np.zeros(self.vector_size, dtype=float)
        word_length = len(word)
        for char in word.lower():
            if char in self.char2index:
                tf = word.count(char) / word_length
                tfidf_vector[self.char2index[char]] = tf * self.idf[char]
        return tfidf_vector
