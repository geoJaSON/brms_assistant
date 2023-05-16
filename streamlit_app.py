#%%
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import pinecone
import streamlit as st
#%%
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_API_ENV = 'northamerica-northeast1-gcp'

#%%
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")
docsearch = Pinecone.from_existing_index('langchain1', embeddings)
#%%


def ask_question():
    docs = docsearch.similarity_search(question)
    return chain.run(input_documents=docs, question=question)


st.header('BRMS Assistant')
st.write('This is a simple app to help you with your BRMS questions')

question = st.text_input('Enter your question here')

if question:
    answer = ask_question()
    st.write(answer)
# %%
 
