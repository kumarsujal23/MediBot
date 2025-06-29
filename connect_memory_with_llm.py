import os, dotenv
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint, 
    HuggingFaceEmbeddings
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

dotenv.load_dotenv()
HF_TOKEN = os.environ["HF_TOKEN"]
REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DB_PATH = r"C:/Users/Lenovo/Desktop/medibot/vectorstore/db_faiss"


base_llm = HuggingFaceEndpoint(
    repo_id=REPO_ID,
    task="conversational",
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=512,
    temperature=0.1,
)

chat_llm = ChatHuggingFace(llm=base_llm)


chat_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a medical assistant. Using the information contained in the context, give a comprehensive answer to the question. 
Respond only to the question asked, response should be concise and relevant to the question.
If the answer cannot be deduced from the context, do not give an answer."""),
    ("user", """Context:
{context}
---
Now here is the question you need to answer.
Question: {question}""")
])


emb_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_PATH, emb_model, allow_dangerous_deserialization=True)


qa_chain = RetrievalQA.from_chain_type(
    llm=chat_llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": chat_prompt}
)


question = input("Ask: ")
answer = qa_chain.invoke({"query": question})
print("RESULT:", answer["result"])


