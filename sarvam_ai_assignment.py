#!/usr/bin/env python
from typing import List

from fastapi import FastAPI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage
from langserve import add_routes


mykey="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"

# 1. Load Retriever
loader = UnstructuredHTMLLoader("/Users/gauravkungwani/Downloads/langchain_crash_course/langchain/toc_notifications_2023_1991/rbi_notification_2023_1991/2023.html")
data = loader.load()

embeddings = OpenAIEmbeddings(openai_api_key=mykey)
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(data)
vector = FAISS.from_documents(documents, embeddings)

retriever = vector.as_retriever()

llm = ChatOpenAI(openai_api_key=mykey)

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever, document_chain)

# 4. App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 5. Adding chain route

# We need to add these input/output schemas because the current AgentExecutor
# is lacking in schemas.

class Input(BaseModel):
    input: str
    chat_history: List[BaseMessage] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "location"}},
    )


class Output(BaseModel):
    output: str

add_routes(
    app,
    retrieval_chain.with_types(input_type=Input, output_type=Output),
    path="/agent",
)

# add_routes(app, retrieval_chain, path="/agent")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)

