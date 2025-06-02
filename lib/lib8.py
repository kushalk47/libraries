from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain import LLMChain



with open("bmw.txt", "r") as f:
    text = f.read()


cohere_api_key = "vGt7iJva8GXi7vcq5U8VD4RmPIPVdjVH8hEZFc0T"

prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in three bullet points:\n{text}"
)

chain = LLMChain(llm=Cohere(cohere_api_key=cohere_api_key), prompt=prompt)
result = chain.run(text)
print("Summary:\n", result)