#%%
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st


#%%pip install chroma
persist_directory = 'db'
embedding = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])

vectordb2 = Chroma(persist_directory=persist_directory, 
                  embedding_function=embedding,
                   )

retriever = vectordb2.as_retriever(search_kwargs={"k": 2})

#%%
# Set up the turbo LLM
turbo_llm = ChatOpenAI(
    temperature=0,
    model_name='gpt-3.5-turbo'
)
# create the chain to answer questions 
qa_chain = RetrievalQA.from_chain_type(llm=turbo_llm, 
                                  chain_type="stuff", 
                                  retriever=retriever, 
                                  return_source_documents=True)
#%%#%%

def ask_question(question):
    try:
        llm_response = qa_chain(question)
        try:
            source = llm_response["source_documents"][0].metadata['source'].replace(r"C:\Users\g3retjjj\Documents\chatbot_data\sop.txt",r"\\nwk-netapp2.nwk.ds.usace.army.mil\MISSIONFILES\MissionProjects\civ\Temporary Roofing\2.0 SOP\2023\2023 Temporary Roofing SOP.pdf")
            return llm_response['result'] + '\n\nSource: ' + source
        except:
            return "Apologies but I cannot find the answer in my source material"
        

    except Exception as e:
        st.error(e)

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
