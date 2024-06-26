import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
from google.api_core.exceptions import Unauthorized as GoogleUnauthorized
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import re
import base64
import sqlite3
import tempfile
import os
import sys
import contextlib
import io
from streamlit_lottie import st_lottie
import requests
import time
import plotly.express as px
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.metric_cards import style_metric_cards

st.set_page_config(page_title="AIDA: AI Data Analyzer", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .quick-insight {
        background-color: #333333;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .quick-insight-header {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .quick-insight-content {
        font-size: 14px;
    }
    .js-plotly-plot .plotly .modebar {
        background-color: #2B2B2B !important;
    }
    .js-plotly-plot .plotly .modebar-btn path {
        fill: #FFFFFF !important;
    }
    .table-container {
        max-width: 100%;
        overflow-x: auto;
        margin-bottom: 1rem;
    }
    .table {
        width: 100%;
        border-collapse: collapse;
        text-align: left;
        margin-bottom: 0;
        background-color: #2B2B2B;
        color: #FFFFFF;
        white-space: nowrap;
    }
    .table th, .table td {
        padding: 0.75rem;
        border-bottom: 1px solid #4A4A4A;
    }
    .table th {
        background-color: #1E1E1E;
        font-weight: bold;
        position: sticky;
        top: 0;
        z-index: 1;
    }
    .table tbody tr:nth-of-type(even) {
        background-color: #333333;
    }
    .table tbody tr:hover {
        background-color: #4A4A4A;
    }
    /* Custom scrollbar styles for Webkit browsers */
    .table-container::-webkit-scrollbar {
        height: 8px;
    }
    .table-container::-webkit-scrollbar-track {
        background: #1E1E1E;
    }
    .table-container::-webkit-scrollbar-thumb {
        background-color: #4A4A4A;
        border-radius: 20px;
        border: 3px solid #1E1E1E;
    }
</style>
""", unsafe_allow_html=True)



def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def create_eda_agent(api_key, model, temperature, max_tokens, df):
    try:
        llm = ChatGoogleGenerativeAI(google_api_key=api_key, model=model, temperature=temperature, max_output_tokens=max_tokens, streaming=True)
    except GoogleUnauthorized:
        st.error("Invalid Gemini API Key.")
        return None
    return create_pandas_dataframe_agent(llm, df, agent_type="tool-calling", verbose=True)

def load_data(file):
    extension = file.name.split('.')[-1]
    if extension == 'csv':
        return pd.read_csv(file)
    elif extension == 'xlsx':
        return pd.read_excel(file)
    elif extension == 'db':
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name
        conn = sqlite3.connect(tmp_file_path)
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        tables = pd.read_sql(query, conn)
        st.write("Tables in database:", tables)
        table_name = st.selectbox("Select a table to load", tables['name'])
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        conn.close()
        os.remove(tmp_file_path)
        return df
    else:
        st.error("Unsupported file type")
        return None
        
@contextlib.contextmanager
def capture_output():
    new_out, new_err = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err

def plot_is_generated(code):
    return "plt" in code or "matplotlib" in code or "plot" in code

def save_plot_as_base64():
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    buf.close()
    plt.clf()
    return img_str

def stream_response(response):
    placeholder = st.empty()
    full_response = ""
    for chunk in response:
        full_response += chunk
        placeholder.markdown(full_response + "▌")
        time.sleep(0.02)
    placeholder.markdown(full_response)
    return full_response

def generate_quick_insights(df):
    insights = []
    
    insights.append({
        "header": "Dataset Overview",
        "content": f"\n📊 Shape: {df.shape[0]} rows, {df.shape[1]} columns"
    })
    
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        insights.append({
            "header": "Missing Values",
            "content": f"🕳️ Columns with missing values:\n{missing_values[missing_values > 0].to_string()}"
        })
    else:
        insights.append({
            "header": "Missing Values",
            "content": "✅ No missing values found in the dataset."
        })
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64'])
    if not numeric_cols.empty:
        stats_df = numeric_cols.describe().transpose()
        stats_df = stats_df.round(2)  # Round to 2 decimal places for clarity
        stats_df = stats_df.reset_index()
        stats_df.columns = ['Column'] + list(stats_df.columns[1:])
        
        table_html = f"""
        <div class="table-container">
            {stats_df.to_html(index=False, classes='table table-striped table-hover')}
        </div>
        """
        
        insights.append({
            "header": "Numeric Columns Statistics",
            "content": table_html,
            "is_table": True
        })
        
    return insights

def main():
    colored_header(
        label="AIDA: AI Data Analyzer 🤖🧑‍💻",
        description="Effortlessly analyze and visualize your data with AI-powered insights using Google's Gemini.",
        color_name="blue-70",
    )

    lottie_url = "https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json"
    lottie_json = load_lottie_url(lottie_url)
    st_lottie(lottie_json, speed=1, height=200, key="initial")

    add_vertical_space(2)

    col1, col2 = st.columns([2, 1])

    with col2:
        st.sidebar.header("Configuration")
        model = st.sidebar.selectbox("Select Model", ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"])
        api_key = st.sidebar.text_input("Enter API Key", key="api_key_input", type='password')
        temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)
        max_tokens = st.sidebar.slider("Max Tokens", 1, 2048, 512)
        clear_history = st.sidebar.button("Clear Chat History")

        if clear_history:
            st.session_state.chat_history = []
        
        st.sidebar.markdown(
            """
            <style>
            .center-text {
                text-align: center;
                color: #FFFFFF;
            }
            .icon-container {
                display: flex;
                justify-content: space-around;
            }
            .icon-container a img {
                width: 32px;
                height: 32px;
                filter: invert(1);
            }
            </style>
            <br><div class="center-text"> Made with ❤️ by Akshat Gupta</div><br>
            <div class="icon-container">
                <a href="https://github.com/akshat122402" target="_blank">
                    <img src="https://img.icons8.com/material-outlined/48/000000/github.png" alt="GitHub"/>
                </a>
                <a href="https://www.kaggle.com/akshatgupta7" target="_blank">
                    <img src="https://img.icons8.com/windows/32/kaggle.png" alt="Kaggle"/>
                </a>
                <a href="https://www.linkedin.com/in/akshat-gupta-a82923227/" target="_blank">
                    <img src="https://img.icons8.com/material-outlined/48/000000/linkedin.png" alt="LinkedIn"/>
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col1:
        st.header("Upload Dataset")
        uploaded_file = st.file_uploader("Upload your data", type=["csv", "xlsx", "db"])

        if uploaded_file:
            df = load_data(uploaded_file)

            if df is not None:
                if 'eda_agent' not in st.session_state:
                    eda_agent = create_eda_agent(api_key, model, temperature, max_tokens, df)
                    if eda_agent:
                        st.session_state.eda_agent = eda_agent
                        st.session_state.chat_history = []

                eda_agent = st.session_state.get('eda_agent', None)

                if eda_agent:
                    st.subheader("Dataset Preview")
                    st.dataframe(df.head(), use_container_width=True)

                    st.subheader("📊 Quick Insights")
                    insights = generate_quick_insights(df)
                    for insight in insights:
                        with st.expander(insight["header"], expanded=True):
                            st.markdown(f'<div class="quick-insight"><div class="quick-insight-content">{insight["content"]}</div></div>', unsafe_allow_html=True)

                    st.subheader("Data Visualization")
                    viz_type = st.selectbox("Select Visualization Type", ["Scatter Plot", "Bar Chart", "Histogram", "Box Plot", "Heatmap"])
                    
                    if viz_type == "Scatter Plot":
                        x_col = st.selectbox("Select X-axis", df.columns)
                        y_col = st.selectbox("Select Y-axis", df.columns)
                        fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                    elif viz_type == "Bar Chart":
                        x_col = st.selectbox("Select X-axis", df.columns)
                        y_col = st.selectbox("Select Y-axis", df.columns)
                        fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                    elif viz_type == "Histogram":
                        col = st.selectbox("Select Column", df.columns)
                        fig = px.histogram(df, x=col, title=f"Histogram of {col}")
                    elif viz_type == "Box Plot":
                        col = st.selectbox("Select Column", df.columns)
                        fig = px.box(df, y=col, title=f"Box Plot of {col}")
                    elif viz_type == "Heatmap":
                        corr = df.corr()
                        fig = px.imshow(corr, title="Correlation Heatmap")
                    
                    fig.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="rgba(0, 0, 0, 0)",
                        paper_bgcolor="rgba(0, 0, 0, 0)",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.header("Chat Interface")

                    for chat in st.session_state.chat_history:
                        if 'type' not in chat:
                            chat['type'] = 'text'  

                    if st.session_state.chat_history:
                        for chat in st.session_state.chat_history:
                            st.markdown(f'<p><img src="https://img.icons8.com/color/48/000000/user.png" width="20"/> <b>You:</b> {chat["user"]}</p>', unsafe_allow_html=True)
                            if chat['type'] == 'text': 
                                st.markdown(f'<p><img src="https://img.icons8.com/fluency/48/000000/robot-3.png" width="20"/> <b>AI:</b> {chat["ai"]}</p>', unsafe_allow_html=True)
                            elif chat['type'] == 'plot':
                                img_bytes = base64.b64decode(chat['ai'])
                                buf = BytesIO(img_bytes)
                                st.image(buf)

                    user_input = st.text_input("You:", key="user_input", placeholder="Ask me anything about your data or generate any chart...")

                    if user_input:
                        response = eda_agent.invoke(user_input)
                        output = stream_response(response['output'])

                        code_pattern = r'```python\n(.*?)```'
                        match = re.search(code_pattern, output, re.DOTALL)

                        if match:
                            code = match.group(1)
                            if plot_is_generated(code):
                                try:
                                    exec(code, globals(), locals())
                                    img_str = save_plot_as_base64()
                                    st.session_state.chat_history.append({"user": user_input, "ai": img_str, "type": "plot"})
                                    st.image(BytesIO(base64.b64decode(img_str)))
                                except Exception as e:
                                    st.error(f"Error generating plot: {e}")
                            else:
                                with capture_output() as (out, err):
                                    exec(code, globals(), locals())
                                exec_output = out.getvalue().strip() or "No Output"
                                st.markdown(f'<p><img src="https://img.icons8.com/color/48/000000/robot-3.png" width="20"/> <b>AI:</b> {exec_output}</p>', unsafe_allow_html=True)
                                st.session_state.chat_history.append({"user": user_input, "ai": exec_output, "type": "text"})
                        else:
                            st.markdown(f'<p><img src="https://img.icons8.com/color/48/000000/robot-3.png" width="20"/> <b>AI:</b> {output}</p>', unsafe_allow_html=True)
                            st.session_state.chat_history.append({"user": user_input, "ai": output, "type": "text"})

if __name__ == "__main__":
    main()
