import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(
    page_title="AI Analytics Assistant",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Analytics Assistant")
st.write("Upload a CSV file and ask questions about your data.")

if not api_key:
    st.error("OPENAI_API_KEY is missing. Please add it to your .env file.")
    st.stop()

client = OpenAI(api_key=api_key)

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

Important:
- You only have access to the dataset summary and first 5 rows.
- Do not claim you calculated totals unless the value is visible in the summary.
- If the question requires full-dataset calculation, say that Pandas analysis is needed.

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

    column_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isnull().sum().values
    })

    st.dataframe(column_info, use_container_width=True)

    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    category_columns = df.select_dtypes(include=["object"]).columns.tolist()

    st.subheader("Quick Analysis")

    if numeric_columns and category_columns:
        selected_category = st.selectbox(
            "Group by category",
            category_columns
        )

        selected_metric = st.selectbox(
            "Select metric",
            numeric_columns
        )

        aggregation = st.selectbox(
            "Select aggregation",
            ["sum", "average", "count", "min", "max"]
        )

        if aggregation == "sum":
            result = df.groupby(selected_category)[selected_metric].sum()
        elif aggregation == "average":
            result = df.groupby(selected_category)[selected_metric].mean()
        elif aggregation == "count":
            result = df.groupby(selected_category)[selected_metric].count()
        elif aggregation == "min":
            result = df.groupby(selected_category)[selected_metric].min()
        else:
            result = df.groupby(selected_category)[selected_metric].max()

        result = (
            result
            .sort_values(ascending=False)
            .reset_index()
        )

        st.dataframe(result, use_container_width=True)

        st.bar_chart(result.set_index(selected_category))

    else:
        st.info("Need at least one text column and one numeric column for quick analysis.")

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

    question = st.chat_input("Example: What cleaning steps would you recommend?")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing your data..."):
                answer = ask_ai(question, df)
                st.write(answer)

else:
    st.info("Upload a CSV file to get started.")