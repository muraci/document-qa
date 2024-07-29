import streamlit as st
import pandas as pd
import sqlite3
from langchain.llms import OpenAI
from langchain.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

# Page configuration
st.set_page_config(page_title="Marketing Campaign Q&A", layout="wide")

# Sidebar for API key input, model selection, temperature, and example questions
with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Enter your OpenAI API key:", type="password")

    st.header("Text Generation Model")
    model = st.selectbox(
        "Choose a model",
        ["gpt-3.5-turbo-instruct", "gpt-3.5-turbo", "gpt-4-turbo"],
        index=0
    )
    
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    
    #use_uploaded_data = st.checkbox("Upload custom data?", value=False)
    
    #if use_uploaded_data:
    #    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    st.header("Example Questions")
    example_questions = [
        "What is the max duration?",
        "What are the unique location names?",
        "What are the unique campaign types?",
        "What is the location with the max ROI?",
        "What is the best performing marketing channel and location according to ROI?",
        "What is the total number of campaigns for each campaign type?",
        "Which is the best performing campaign?"
    ]
    selected_question = st.selectbox("Select an example question:", [""] + example_questions)

# Main content
st.title("Marketing Campaign Q&A")

# Function to load data and set up database
@st.cache_resource
def load_data(file):
    if isinstance(file, str) and file.startswith('http'):
        df = pd.read_csv(file)
    elif hasattr(file, 'read'):  # Check if it's a file-like object (UploadedFile)
        df = pd.read_csv(file)
    else:
        raise ValueError("Invalid input. Expected a URL string or an uploaded file.")
    
    conn = sqlite3.connect('Marketing.sqlite')
    df.to_sql('Marketing', conn, if_exists='replace', index=False)
    conn.close()
    return SQLDatabase.from_uri('sqlite:///Marketing.sqlite')

# Load data
if use_uploaded_data and uploaded_file is not None:
    input_db = load_data(uploaded_file)
else:
    input_db = load_data("https://raw.githubusercontent.com/muraci/document-qa/main/marketing_campaign_data.csv")

# Function to get the first 5 rows from SQLite database
def get_first_5_rows():
    with sqlite3.connect('Marketing.sqlite') as conn:
        return pd.read_sql_query("SELECT * FROM Marketing LIMIT 5", conn)

# Display the first 5 rows of the database
st.subheader("Database Preview (First 5 rows from SQLite)")
st.dataframe(get_first_5_rows())

# Set up OpenAI LLM and SQLDatabaseChain
@st.cache_resource
def setup_agent(_api_key, _model, _temperature):
    if not _api_key:
        return None
    llm = OpenAI(temperature=_temperature, api_key=_api_key, model_name=_model)
    return SQLDatabaseChain(llm=llm, database=input_db, verbose=True)

# Main app logic
if api_key:
    db_agent = setup_agent(api_key, model, temperature)
    if db_agent:
        st.success("API key set and database loaded successfully!")
        
        # User input for question
        user_question = st.text_input("Ask a question about the marketing campaign data:", value=selected_question)
        
        if user_question:
            if st.button("Get Answer"):
                try:
                    with st.spinner("Generating answer..."):
                        result = db_agent.run(user_question)
                    st.subheader("Answer:")
                    st.write(result)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    else:
        st.error("Failed to set up the database agent. Please check your API key.")
else:
    st.warning("Please enter your OpenAI API key in the sidebar to proceed.")

# Additional information
st.markdown("---")
st.info("This app uses OpenAI's language model to answer questions about marketing campaign data. "
        "Enter your API key, select a model and temperature, choose or upload data, select an example question or type your own, and click 'Get Answer' to see the results.")
