from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnableSequence

load_dotenv()

prompt = PromptTemplate(
    template="Give me the creative joke based on the {topic}",
    input_variables=["topic"]
)

model = ChatOpenAI(model = 'gpt-5.2')

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="give the explination of the given Joke {joke}",
    input_variables=["joke"]
)

runnable = RunnableSequence(prompt, model, parser, prompt1, model, parser)

result = runnable.invoke({"topic": "Politics"})

print(result)
