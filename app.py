import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="AI Analytics Assistant",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Analytics Assistant")
st.write("Upload campaign data and explore it instantly.")


st.write("analytics")


def clean_column_names(df):
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("CSV uploaded successfully.")
else:
    df = pd.read_csv("sample_data/campaigns.csv")
    st.info("Using sample campaign dataset.")

df = clean_column_names(df)

st.subheader("Data Preview")
st.dataframe(df.head(20), use_container_width=True)

st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rows", df.shape[0])

with col2:
    st.metric("Columns", df.shape[1])

with col3:
    st.metric("Missing Values", int(df.isna().sum().sum()))

st.subheader("Column Details")

column_info = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str).values,
    "Non-Null Count": df.notnull().sum().values,
    "Missing Count": df.isnull().sum().values
})

st.dataframe(column_info, use_container_width=True)

st.subheader("Basic Numeric Statistics")

numeric_df = df.select_dtypes(include="number")

if not numeric_df.empty:
    st.dataframe(numeric_df.describe(), use_container_width=True)
else:
    st.warning("No numeric columns found.")

st.subheader("Available Columns")
st.write(", ".join(df.columns))