import streamlit as st
import pandas as pd
import sqlite3
from langchain.llms import OpenAI
from langchain.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
import requests
from io import StringIO

# Page configuration
st.set_page_config(page_title="Marketing Campaign Q&A", layout="wide")

# CSV URL
CSV_URL = "https://drive.google.com/file/d/1o2hNhKiUXVP50zldEbpnKxfvurtdnjkD/view?usp=sharing"

# Function to load data and set up database
@st.cache_resource
def load_data(delimiter):
    try:
        # Fetch CSV content
        response = requests.get(CSV_URL)
        response.raise_for_status()  # Raise an exception for bad responses
        csv_content = response.text

        # Preview CSV content
        st.subheader("CSV Content Preview")
        st.text(csv_content[:500] + "...")  # Show first 500 characters

        # Try to parse CSV
        df = pd.read_csv(StringIO(csv_content), delimiter=delimiter)

        # Preview DataFrame
        st.subheader("DataFrame Preview")
        st.dataframe(df.head())

        # Display DataFrame info
        st.subheader("DataFrame Info")
        buffer = StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        # Create SQLite database
        conn = sqlite3.connect('Marketing.sqlite')
        df.to_sql('Marketing', conn, if_exists='replace', index=False)
        conn.close()

        return SQLDatabase.from_uri('sqlite:///Marketing.sqlite'), df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

# Sidebar for settings and example questions
with st.sidebar:
    st.title("Settings & Examples")
    api_key = st.text_input("Enter your OpenAI API key:", type="password")
    
    delimiter = st.selectbox("Select CSV delimiter:", [",", ";", "|", "\t"], index=0)
    
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

# Load data
if st.button("Load Data"):
    input_db, df = load_data(delimiter)

    if input_db is not None and df is not None:
        # Set up OpenAI LLM and SQLDatabaseChain
        @st.cache_resource
        def setup_agent(_api_key):
            if not _api_key:
                return None
            llm = OpenAI(temperature=0, api_key=_api_key)
            return SQLDatabaseChain(llm=llm, database=input_db, verbose=True)

        # Main app logic
        if api_key:
            db_agent = setup_agent(api_key)
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
    else:
        st.error("Failed to load the data. Please check the CSV file and try again.")

# Additional information
st.markdown("---")
st.info("This app uses OpenAI's language model to answer questions about marketing campaign data. "
        "Enter your API key, select the correct CSV delimiter, load the data, then select an example question or type your own, and click 'Get Answer' to see the results.")
