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
docsearch = Pinecone.from_existing_index('esfsearch', embeddings)
#%%

css = """
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #94e5ff
}
.chat-message.bot {
    background-color: #FF7276
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
</style>
"""

bot_template = """
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://cdn.dribbble.com/users/160155/screenshots/1526505/media/617bae5c11b9b42021065f5a610001dc.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
"""
user_template = """
<div class="chat-message user">
    <div class="avatar">
        <img src="https://www.publications.usace.army.mil/Portals/76/Publications/EngineerStandardsGraphics/gs-15.gif">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
"""
st.markdown(css, unsafe_allow_html=True)


def ask_question(question):
    try:
        docs = docsearch.similarity_search(question)
        return chain.run(input_documents=docs, question=question)
    except Exception as e:
        st.error("An error has occurred. Please try again.")

# Initialize session state if not yet done
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.sidebar.header('ESF #3 Chatbot BETA')
st.sidebar.image("uCOP_logo.png", use_column_width=True)  # add logo
question = st.text_input('Enter your question here')

if question and (len(st.session_state.chat_history) == 0 or question != st.session_state.chat_history[-1][0]):
    with st.spinner('Thinking...'):
        answer = ask_question(question)
        # Save question and answer in the session state
        st.session_state.chat_history.append((question, answer))

# Display chat history

for q, a in reversed(st.session_state.chat_history):
    st.write(user_template.replace("{{MSG}}", q), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", a), unsafe_allow_html=True)


# %%
