from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv()

def word_count(text: str):
    return len(text.split())

prompt = PromptTemplate(
    template="Give me the creative joke based on the {topic}",
    input_variables=["topic"]
)

model = ChatOpenAI(model='gpt-5.2')

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt, model, parser)

joke_word_count_parallel = RunnableParallel({
    "Joke": RunnablePassthrough(),
    "Num_of_Words": RunnableLambda(word_count)
})

final_chain = RunnableSequence(joke_gen_chain, joke_word_count_parallel)

result = final_chain.invoke({"topic":"AI Revolution"})

print(result["Joke"])
print("total number of words : ",result["Num_of_Words"])