from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableBranch, RunnableLambda

load_dotenv()

prompt1 = PromptTemplate(
    template="Generate the report on {topic}, act as professional writer",
    input_variables= ["topic"]
)

model = ChatOpenAI(model='gpt-5.2')

parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt1, model, parser)

prompt2 = PromptTemplate(
    template="Summarize the report {report}",
    input_variables= ["report"]
)

condt_result_chain = RunnableBranch(
    (lambda x:len(x.split()) > 300,RunnableSequence(prompt2, model, parser) ),
    RunnablePassthrough()
)

report_word_cnt_parallel = RunnableParallel(
    {
     "Report": RunnablePassthrough(),
     "Word_Count": RunnableLambda(lambda x: len(x.split()))
    }
)

final_chain = RunnableSequence(report_gen_chain, condt_result_chain, report_word_cnt_parallel)

result = final_chain.invoke({"topic": "RAG based App"})

print(result["Report"])
print("Word Count : ",result["Word_Count"])
