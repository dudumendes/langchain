from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
import os
from langchain.globals import set_debug

set_debug(True)

llm = ChatOpenAI(
  model="gpt-4-0125-preview", 
  temperature=0.5, 
  api_key=os.getenv("OPENAI_API_KEY"))

examples = [
    {
        "question": "Quem viveu mais, Muhammad Ali ou Alan Turing?",
        "answer": """
        São necessárias perguntas de acompanhamento: Sim.
        Pergunta: Quantos anos Muhammad Ali tinha quando morreu?
        Resposta intermediária: Muhammad Ali tinha 74 anos quando morreu.
        Pergunta: Quantos anos Alan Turing tinha quando morreu?
        Resposta intermediária: Alan Turing tinha 41 anos quando morreu.
        Então a resposta final é: Muhammad Ali
        """,
    },
    # Outros exemplos aqui...
]

example_prompt = PromptTemplate(
    input_variables=["question", "answer"], template="Pergunta: {question}\n{answer}"
)

prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Pergunta: {input}",
    input_variables=["input"],
)

prompt = prompt_template.format(input="Quem foi o pai do Rei Charles atual?")
resposta = llm.invoke(prompt)
print(resposta)