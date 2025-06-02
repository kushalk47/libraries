#key components to remeber tsne(n_components,perplexity=5,random_state=42)
#tsne.fit_transformer()
#creating a numpy array after getting the vectors
# word_vector comes under keyedvector and we can directly pass the vector to the model initialized
# the s parameter is kinda important 

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.downloader import load
import numpy as np

# Load pre-trained word vectors (GloVe - 100 dimensions)
word_vectors = load('glove-wiki-gigaword-100')

# Select 10 words from the "technology" domain (filtering directly)
tech_words = ['computer', 'internet', 'software', 'hardware', 'network', 
              'data', 'cloud', 'robot', 'algorithm', 'technology']

# Filter words present in the model
filtered_tech_words = [word for word in tech_words if word in word_vectors]

# Extract vectors for filtered words
vectors = np.array([word_vectors[word] for word in filtered_tech_words])

# Reduce dimensions using t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
reduced_vectors = tsne.fit_transform(vectors)

# Simple scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], color='skyblue', edgecolors='black', s=100)

# Add labels directly
for i, word in enumerate(filtered_tech_words):
    plt.text(reduced_vectors[i, 0]+0.02, reduced_vectors[i, 1]+0.02, word, fontsize=12)

plt.title("t-SNE Visualization of Technology Words", fontsize=14)
plt.xticks([])  
plt.yticks([])
plt.show()


input_word = 'computer'
try:
    similar_words = word_vectors.most_similar(input_word, topn=5)
    print(f"\n5 words similar to '{input_word}':")
    for word, similarity in similar_words:
        print(f"{word} (similarity: {similarity:.2f})")
except KeyError:
    print(f"'{input_word}' is not in the vocabulary.")
