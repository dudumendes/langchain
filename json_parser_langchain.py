from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.globals import set_debug
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
import os
from dotenv import load_dotenv

# set_debug(True)
load_dotenv()

class Destination(BaseModel):
    city: str = Field(description="Nome da cidade a visitar")
    motivation: str = Field(description="Motivo para visitar a cidade")
    restaurants: str = Field(description="Restaurantes populares na cidade")
    culture: str = Field(description="Atividades culturais populares na cidade")

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

parser = JsonOutputParser(pydantic_object=Destination)

city_template = PromptTemplate(
    template = """Sugira uma cidade dado meu interesse por {interesse}.
      {formatacao_de_saida}
    """,
    input_variables=["interesse"],
    partial_variables={"formatacao_de_saida": parser.get_format_instructions()}
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
