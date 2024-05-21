import streamlit as st
from langchain import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

import warnings
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings('ignore')

loader = TextLoader('documents/companies_info.txt', autodetect_encoding=True)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
store = Chroma.from_documents(texts, embeddings, collection_name="state-of-the-union")
llm = OpenAI(temperature=0.7)
chain = RetrievalQA.from_chain_type(llm, retriever=store.as_retriever())

def chat():
    st.write("Chat bot based on salesforce data")

    question = st.text_input("Ask a question:")
    if st.button("Submit"):
        answer = get_answer(question)
        st.write("Answer: ", answer)


def get_answer(question):

    result = chain.run(question)

    return (result)


if __name__ == "__main__":
    image = "src/chatsales.png"  # Replace with your image URL
    st.image(image, use_column_width=True)
    chat()





