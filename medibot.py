import os, dotenv
import streamlit as st

from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFaceEmbeddings
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA


st.set_page_config(
    page_title="MediBot - AI Medical Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
    /* Main background and font */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #333;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 15px !important;
        padding: 15px !important;
        margin: 10px 0 !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        border-left: 4px solid #3498db !important;
        color: #333 !important;
    }
    
    /* Chat message text color fix */
    .stChatMessage p, .stChatMessage div {
        color: #333 !important;
    }
    
    /* Title styling */
    .main-title {
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #e74c3c, #c0392b) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 10px 20px !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(52, 152, 219, 0.1);
        border-left: 4px solid #3498db;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        color: #fff !important;
    }
    
    /* Welcome card styling */
    .welcome-card {
        background: rgba(255, 255, 255, 0.9);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: #333 !important;
    }
    
    /* Chat input styling - FIX FOR WHITE TEXT */
    .stChatInput > div > div > input {
        border-radius: 25px !important;
        border: 2px solid #3498db !important;
        padding: 15px 20px !important;
        font-size: 1rem !important;
        background-color: #fff !important;
        color: #333 !important;
    }
    
    .stChatInput > div > div > input::placeholder {
        color: #666 !important;
    }
    
    /* Sidebar text color fix */
    .css-1d391kg {
        color: #fff !important;
    }
    
    /* Widget labels color fix */
    [data-testid="stWidgetLabel"] p {
        color: #fff !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #f39c12, #e67e22) !important;
        color: white !important;
        border-radius: 10px !important;
        font-weight: bold !important;
    }
    
    /* Main content text color */
    .main .block-container {
        color: #fff !important;
    }
    
    /* Example button styling */
    .example-question {
        background: linear-gradient(45deg, #27ae60, #2ecc71) !important;
        color: white !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 10px 15px !important;
        margin: 5px !important;
        font-size: 0.9rem !important;
        transition: all 0.3s ease !important;
    }
    
    /* Fix for metric labels */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        padding: 10px !important;
    }
    
    [data-testid="metric-container"] > div {
        color: #fff !important;
    }
</style>
""", unsafe_allow_html=True)

dotenv.load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DB_FAISS_PATH = r"vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    """Loads the FAISS vector store from the local path."""
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a medical assistant with STRICT LIMITATIONS. You can ONLY answer questions using the medical information provided in the context below.

CRITICAL RULES - FOLLOW EXACTLY:
1. If the context does not contain information to answer the question, you MUST respond EXACTLY with: "Could not assist you with information as question is out of scope of mine"
2. NEVER use your general knowledge - ONLY use the provided context
3. If the question is about non-medical topics (sports, entertainment, general knowledge, etc.), respond EXACTLY with: "Could not assist you with information as question is out of scope of mine"
4. Do NOT explain why you can't answer - just give the exact response above
5. IGNORE all previous conversations - treat each question independently
6. Only answer if the context contains relevant medical information for the specific question asked

Context: {context}"""),
        ("user", "Question: {question}")
    ])
    return prompt

def load_llm():
    """Loads the Hugging Face model and wraps it in the ChatHuggingFace class."""
    base_llm = HuggingFaceEndpoint(
        repo_id=REPO_ID,
        task="conversational",
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512,
        temperature=0.1,
    )
    return ChatHuggingFace(llm=base_llm)

def main():
    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown("# 🩺  MediBot")
        st.markdown("---")
        
        st.markdown("""
        <div class="info-box">
            <h3>🩺 About MediBot</h3>
            <p>Your intelligent medical assistant powered by advanced AI technology. Get evidence-based answers to your health questions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h4>✨ Features:</h4>
            <ul>
                <li>🔍 Evidence-based responses</li>
                <li>📚 Source document references</li>
                <li>🤖 AI-powered insights</li>
                <li>🔒 Privacy focused</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("""
        <div style="background: rgba(231, 76, 60, 0.1); padding: 15px; border-radius: 10px; margin-top: 20px;">
            <h4 style="color: #fff;">⚠️ Medical Disclaimer</h4>
            <p style="font-size: 0.8rem; color: #fff;">This AI assistant provides general information only. Always consult qualified healthcare professionals for medical advice, diagnosis, or treatment.</p>
        </div>
        """, unsafe_allow_html=True)

    
    st.markdown('<h1 class="main-title">🩺 MediBot - Your AI Medical Assistant</h1>', unsafe_allow_html=True)

    
    if "messages" not in st.session_state or len(st.session_state.messages) <= 1:
        st.markdown("""
        <div class="welcome-card">
            <h2>👋 Welcome to MediBot!</h2>
            <p style="font-size: 1.1rem; color: #555;">Ask me anything about health, medicine, symptoms, or treatments. I'm here to provide evidence-based information.</p>
        </div>
        """, unsafe_allow_html=True)
        
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! 👋 I'm MediBot, your AI medical assistant. How can I help you with your health questions today?"}
        ]

    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

 
    if prompt := st.chat_input("💬 Ask your medical question here..."):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

     
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing your question..."):
                try:
                    vectorstore = get_vectorstore()
                    if vectorstore is None:
                        st.error("Failed to load the knowledge base")
                        return

                
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=load_llm(),
                        chain_type="stuff",
                        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                        return_source_documents=True,
                        chain_type_kwargs={"prompt": set_custom_prompt()}
                    )

                  
                    response = qa_chain.invoke({'query': prompt})
                    result = response["result"]
                    
                    st.markdown(result)

                    

                    st.session_state.messages.append({"role": "assistant", "content": result})

                except Exception as e:
                    error_message = f"❌ Sorry, I encountered an error: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    main()
