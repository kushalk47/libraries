from transformers import pipeline

summarizer = pipeline("summarization")

long_text = """
Artificial Intelligence (AI) is transforming various industries by automating tasks, improving
efficiency,
and enabling new capabilities. In the healthcare sector, AI is used for disease diagnosis,
personalized medicine,
and drug discovery. In the business world, AI-powered systems are optimizing customer
service, fraud detection,
and supply chain management. AI's impact on everyday life is significant, from smart
assistants to recommendation
systems in streaming platforms. As AI continues to evolve, it promises even greater
advancements in fields like
education, transportation, and environmental sustainability.
"""

summary = summarizer(long_text, max_length=20, min_length=20,
                     do_sample=False)[0]["summary_text"]

print("Summarized Text:")
print(summary)