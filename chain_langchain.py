from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.globals import set_debug

import os
from dotenv import load_dotenv
set_debug(True)

load_dotenv()

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

city_template = ChatPromptTemplate.from_template(
    "Sugira uma cidade dado meu interesse por {interesse}." \
    "A saída deve ser apenas o nome da cidade, sem explicações ou contexto adicional." \
)

restaurant_template = ChatPromptTemplate.from_template(
  "Sugira restaurantes populares entre locais em {cidade}."
)

culture_template = ChatPromptTemplate.from_template(
    "Sugira atividades culturais populares entre locais em {cidade}."
)

city_chain = LLMChain(prompt=city_template, llm=llm)
restaurant_chain = LLMChain(prompt=restaurant_template, llm=llm) 
culture_chain = LLMChain(prompt=culture_template, llm=llm)

chain = SimpleSequentialChain(
    chains=[city_chain, restaurant_chain, culture_chain],
    verbose=True
)

result = chain.invoke("praias")

print(result)

# The above code creates a sequential chain of prompts to suggest a city, restaurants, and cultural activities based on the user's interest in "praia" (beach).
# The final output will be a structured response containing the suggested city, popular restaurants, and cultural activities in that city.
