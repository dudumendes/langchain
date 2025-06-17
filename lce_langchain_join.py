from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from operator import itemgetter
from langchain.globals import set_debug
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
  "Sugira restaurantes populares entre locais em {city}."
)

culture_template = ChatPromptTemplate.from_template(
    "Sugira atividades culturais populares entre locais em {city}."
)

final_template = ChatPromptTemplate.from_messages(
    [
        ("ai", "Sugestão de viagem para a cidade : {city}"),
        ("ai", "Restaurantes imperdíveis: {restaurants}"),
        ("ai", "Atividades culturais recomendadas: {culture}"),
        ("system", "Combine as informacoes das anteriores em 2 paragraos coerentes.")
    ]
)

part1 = city_template | llm | parser 
part2 = restaurant_template | llm | StrOutputParser()
part3 = culture_template | llm | StrOutputParser() 
part4 = final_template | llm | StrOutputParser()

chain = (part1 
        | { 
            "restaurants" : part2,
            "culture" : part3,
            "city" : itemgetter("city")
          }
        | part4)

result = chain.invoke({"interesse": "praias"})
print(result)