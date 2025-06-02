#1.join if similar words are there else only join the word , so before joining lower and split the words etc
# remeber to put lower , close the f string , default parameters should be at the last 
import gensim.downloader as api

print("Loading pre-trained GloVe model")
embedding_model = api.load('glove-wiki-gigaword-100')
print("Model loaded.")

def get_similar_words(word, model, topn=2):
    clean_word = word.lower().strip(".,!?")
    if clean_word in model.key_to_index:
        return [term for term, _ in model.most_similar(clean_word, topn=topn)]
    return []

def enrich_prompt(prompt, model, topn_similar=2):
    words = prompt.split()
    enriched_parts = []
    for word in words:
        similar_terms = get_similar_words(word, model, topn_similar)
        if similar_terms:
            enriched_parts.append(f"{word} ({', '.join(similar_terms)})")
             
        else:
            enriched_parts.append(word)
    return " ".join(enriched_parts)

original_prompt = "what is science ."
print(f"\nOriginal Prompt: {original_prompt}")

enriched_test_prompt = enrich_prompt(original_prompt, embedding_model, topn_similar=2)
print(f"Enriched Prompt: {enriched_test_prompt}")
