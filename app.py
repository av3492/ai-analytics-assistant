import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(
    page_title="AI Analytics Assistant",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Analytics Assistant")
st.write("Upload a CSV file and ask questions about your data.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

def get_dataframe_summary(df: pd.DataFrame) -> str:
    summary = f"""
Dataset Summary:
Rows: {df.shape[0]}
Columns: {df.shape[1]}

Column Names:
{list(df.columns)}

Data Types:
{df.dtypes.to_string()}

Missing Values:
{df.isnull().sum().to_string()}

First 5 Rows:
{df.head().to_string()}
"""
    return summary

def ask_ai(question: str, df: pd.DataFrame) -> str:
    summary = get_dataframe_summary(df)

    prompt = f"""
You are an expert data analyst.

Use the dataset summary below to answer the user's question.
Be clear, practical, and concise.
If calculations are needed, explain the logic.

{summary}

User question:
{question}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("File uploaded successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Basic Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Missing Values", int(df.isnull().sum().sum()))

    st.subheader("Column Data Types")
    st.dataframe(
        pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isnull().sum().values
        }),
        use_container_width=True
    )

    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if numeric_columns:
        st.subheader("Quick Chart")

        selected_column = st.selectbox(
            "Select a numeric column to visualize",
            numeric_columns
        )

        fig, ax = plt.subplots()
        df[selected_column].plot(kind="bar", ax=ax)
        ax.set_title(f"{selected_column} Values")
        ax.set_xlabel("Row Number")
        ax.set_ylabel(selected_column)
        st.pyplot(fig)

    st.subheader("Ask AI About Your Data")

    question = st.chat_input("Example: Which product has the highest sales?")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing your data..."):
                answer = ask_ai(question, df)
                st.write(answer)

else:
    st.info("Upload a CSV file to get started.")