from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.globals import set_debug
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv

# set_debug(True)
load_dotenv()

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

part1 = PromptTemplate.from_template("Analisar a queixa : {complaint}") | llm | StrOutputParser()
part2 = PromptTemplate.from_template("Avaliar o sentimento da queixa : {analysis_result}") | llm | StrOutputParser()
part3 = PromptTemplate.from_template("Formular resposta: {sentiment}") | llm | StrOutputParser()
 
chain = ({"complaint": RunnablePassthrough()}
          | RunnablePassthrough.assign(analysis_result=part1) 
          | RunnablePassthrough.assign(sentiment=part2)
          | part3
          )

complaint_text = "A comida estava fria e o serviço foi muito lento."

result = chain.invoke({"complaint": complaint_text})
print(result)