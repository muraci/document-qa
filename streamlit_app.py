import os
import numpy as np
import pandas as pd
import sqlite3
import streamlit as st
from langchain.llms import OpenAI
from langchain.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
import warnings
warnings.filterwarnings('ignore')

# Streamlit app
st.title("Marketing Campaign Q&A")

# Get OpenAI API key from user
api_key = st.text_input("Enter your OpenAI API key:", type="password")
os.environ['OPENAI_API_KEY'] = api_key

# Function to load data and set up database
@st.cache_resource
def load_data():
    # Read CSV file
    df = pd.read_csv("https://raw.githubusercontent.com/muraci/document-qa/main/marketing_campaign.csv?token=GHSAT0AAAAAACVC7FYNPDZ6UJFOX6DQWMPIZVHL7CA")
    
    # Create SQL database from the CSV file
    conn = sqlite3.connect('Marketing.sqlite')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS Marketing (
        Campaign_ID TEXT,
        Company TEXT,
        Campaign_Type TEXT,
        Target_Audience TEXT,
        Duration TEXT,
        Channel_Used TEXT,
        Conversion_Rate TEXT,
        Acquisition_Cost TEXT,
        ROI TEXT,
        Location TEXT,
        Language TEXT,
        Clicks TEXT,
        Impressions TEXT,
        Engagement_Score TEXT,
        Customer_Segment TEXT,
        Date TEXT
    )''')
    conn.commit()
    df.to_sql('Marketing', conn, if_exists='replace', index=False)
    
    return SQLDatabase.from_uri('sqlite:///Marketing.sqlite')

# Load data and set up database
input_db = load_data()

c.execute('''SELECT * FROM Marketing''')
for row in c.fetchall():
    st.write(row)

# Set up OpenAI LLM and SQLDatabaseChain
@st.cache_resource
def setup_agent(_api_key):
    if not _api_key:
        return None
    llm_1 = OpenAI(temperature=0)
    return SQLDatabaseChain(llm=llm_1, database=input_db, verbose=True)

# Main app logic
if api_key:
    db_agent = setup_agent(api_key)
    if db_agent:
        st.success("API key set and database loaded successfully!")
        
        # User input for question
        user_question = st.text_input("Ask a question about the marketing campaign data:")
        
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
    st.warning("Please enter your OpenAI API key to proceed.")

# Optional: Display some example questions
st.sidebar.header("Example Questions")
example_questions = [
    "What is the max duration?",
    "What are the unique location names?",
    "What are the unique campaign types?",
    "What is the location with the max ROI?",
    "What is the best performing marketing channel and location according to ROI?",
    "What is the total number of campaigns for each campaign type?",
    "Which is the best performing campaign?"
]
st.sidebar.write("\n".join(f"- {q}" for q in example_questions))
