from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnableParallel, RunnableSequence

load_dotenv()

prompt1 = PromptTemplate(
    template="Give me the tweet to post based on the {topic}",
    input_variables=["topic"]
) 

prompt2 = PromptTemplate(
    template="Give me the Linkedin post based on the {topic}",
    input_variables=["topic"]
) 

model = ChatOpenAI(model='gpt-5.2')

parser = StrOutputParser()

runnable = RunnableParallel({
    "tweet": RunnableSequence(prompt1, model, parser),
    "linkedin": RunnableSequence(prompt2, model, parser)
})

result = runnable.invoke({"topic": "AI Agents"})

print("Tweet: ",result["tweet"])
print("Linked In: ",result["linkedin"])