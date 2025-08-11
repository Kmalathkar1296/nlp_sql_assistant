import os
import sys
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass


import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from langchain_community.vectorstores import Chroma
from langchain_community.utilities import SQLDatabase
from langchain_community.embeddings import OpenAIEmbeddings
from sqlalchemy import create_engine, text
os.environ["CHROMA_DB_IMPL"] = "duckdb+parquet" if not os.environ.get("STREAMLIT_RUNTIME") else "duckdb+parquet"
import chromadb

# ---------------------
# CONFIG
# ---------------------
st.set_page_config(page_title="RAG-Powered Text-to-SQL Assistant", layout="wide")
openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.warning("Please set your OpenAI API key in Streamlit secrets or environment variable.")
    st.stop()

# ---------------------
# DB CONNECTION
# ---------------------
db_path = "sqlite:///university.db"
db = SQLDatabase.from_uri(db_path)

# ---------------------
# VECTOR STORE (Schema as RAG context)
# ---------------------
schema_text = db.get_table_info()

chroma_client = chromadb.Client(
    chromadb.config.Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=None  # Memory-only mode
    )
)
embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])

vectorstore = Chroma.from_texts(
    [schema_text],
    embeddings,
    collection_name="nlp_sql_assistant",
    client=chroma_client
)
retriever = vectorstore.as_retriever()

# ---------------------
# LLM
# ---------------------
llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=openai_api_key)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant that converts natural language to SQL.
Use the schema provided to create the query.

Schema:
{context}

Question:
{question}

SQL Query:
"""
)

# ---------------------
# FUNCTIONS
# ---------------------
def generate_sql(question):
    context = retriever.get_relevant_documents(question)[0].page_content
    final_prompt = prompt.format(context=context, question=question)
    sql_query = llm.predict(final_prompt)
    return sql_query.strip()

def run_query(sql_query):
    try:
        engine = create_engine(db_path)
        with engine.connect() as conn:
            result = conn.execute(text(sql_query)).fetchall()
            return result
    except Exception as e:
        return f"Error: {e}"

def explain_results(question, results):
    explanation_prompt = f"""
Question: {question}
Results: {results}

Explain the results in simple terms:
"""
    return llm.predict(explanation_prompt)

# ---------------------
# UI
# ---------------------
st.title("💬 RAG-Powered LLM: Text-to-SQL Assistant")
st.markdown("Ask questions in plain English, get SQL + results instantly.")

question = st.text_input("Enter your question:", placeholder="e.g. Show top 3 students in Computer Science by GPA")

if st.button("Run Query"):
    if question:
        with st.spinner("Generating SQL..."):
            sql_query = generate_sql(question)
            st.code(sql_query, language="sql")

        with st.spinner("Executing query..."):
            results = run_query(sql_query)
            st.write("### Query Results:")
            st.write(results)

        with st.spinner("Explaining results..."):
            explanation = explain_results(question, results)
            st.write("### Explanation:")
            st.write(explanation)
    else:
        st.warning("Please enter a question.")
