import google.generativeai as genai

genai.configure(os.getnenv(api_key))
model = genai.GenerativeModel('gemini-1.5-flash')

try:
    with open("q.txt", 'r') as f:
        rule = f.read().strip()
    if not rule: raise ValueError("IPC rule is empty.")
except Exception as e:
    print(f"Error: {e}"); exit()

def ask(q):
    prompt = f"Indian Penal Code: '{rule}'\nAnswer: {q}"
    try: return model.generate_content(prompt).text
    except Exception as e: return f"Error: {e}"

print("Ask about IPC (type 'exit' to quit).")
while True:
    q = input("> ")
    if q.lower() == 'exit': break
    print(ask(q))
