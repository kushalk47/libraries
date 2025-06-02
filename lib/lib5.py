import gensim.downloader as api

print("Loading GloVe model (glove-wiki-gigaword-100)... This might take a moment.")
embedding_model = api.load('glove-wiki-gigaword-100')
print("Model loaded.")

def get_similar_words_from_embeddings(seed_word, model, topn=3):
    clean_word = seed_word.lower().strip(".,!?")
    if clean_word in model.key_to_index:
        similar_terms_with_scores = model.most_similar(clean_word, topn=topn)
        return [term for term, _ in similar_terms_with_scores]
    return []

def create_paragraph_with_embeddings(seed_word, model):
    similar_words = get_similar_words_from_embeddings(seed_word, model, topn=2)

    if not similar_words:
        return f"Sorry, I couldn't find similar words for '{seed_word}' in the model's vocabulary."

    paragraph = (
        f"Once upon a time, a grand {seed_word} unfolded. "
        f"It involved deep {similar_words[0]} and exciting {similar_words[1]}. "
        f"Everyone who witnessed this amazing {seed_word} was captivated by its scope."
    )
    return paragraph

seed_word = "Tournament"

print(f"\nSeed word: '{seed_word}'")
story = create_paragraph_with_embeddings(seed_word, embedding_model)
print("Generated Paragraph:")
print(story)