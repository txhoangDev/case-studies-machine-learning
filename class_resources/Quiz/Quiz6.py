import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')

text = "I waited for the train. The train was late. Dr. Jiao and Seung Jun took the bus. I looked for Seung Jun and Dr. Jiao at the bus station."

result = [word_tokenize(sent) for sent in sent_tokenize(text)]
print(result[2][7])

sentences = ["The quick brown fox jumps over the lazy dog.", "This is another sentence."]
for sentence in sentences:
    words = nltk.word_tokenize(sentence)
    print(words)


import numpy as np
from numpy.linalg import norm
vec_A = [1,2,3,4,5]
vec_B = [1,3,5,7,9]

similarity = np.dot(vec_A, vec_B) / (norm(vec_A) * norm(vec_B))
print(similarity)
