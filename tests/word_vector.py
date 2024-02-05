import deeprai.embedding.word_vectorize as x

corpus = ["hello", "world", "apple", "banana", "grape", "orange"]
vectorizer = x.WordVectorizer(corpus)

word = "hello"

print("One-Hot Encoding of 'hello':")
print(vectorizer.one_hot_vectorize(word))
print("\nContinuous Encoding of 'hello':")
print(vectorizer.continuous_vectorize(word))
print("\nBinary Encoding (using ASCII values) of 'hello':")
print(vectorizer.binary_vectorize(word))
print("\nFrequency Encoding of 'hello':")
print(vectorizer.frequency_vectorize(word))
print("\nN-Gram Encoding of 'hello':")
print(vectorizer.ngram_vectorize(word))
print("\nTF-IDF Encoding of 'hello':")
print(vectorizer.tfidf_vectorize(word))