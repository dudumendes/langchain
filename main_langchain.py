from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

numero_de_dias = 7
numero_de_criancas = 2
atividade = "praia"

template = PromptTemplate.from_template(
    "Crie um roteiro de viagem de {numero_de_dias} dias, para uma família com {numero_de_criancas} crianças que gostam de {atividade}"
)

prompt = template.format(
    numero_de_dias=numero_de_dias,
    numero_de_criancas=numero_de_criancas,
    atividade=atividade
)

# prompt = f"Crie um roteiro de viagem de {numero_de_dias} dias, para uma família com {numero_de_criancas} crianças que gostam de {atividade}"
print(prompt)

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

response = llm.invoke(prompt)
print(response.content)