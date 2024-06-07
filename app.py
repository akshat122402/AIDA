import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
from google.api_core.exceptions import Unauthorized as GoogleUnauthorized
import matplotlib.pyplot as plt
from io import BytesIO
import re
import base64
import sqlite3
import tempfile
import os
import sys
import contextlib
import io

def create_eda_agent(api_key, model, temperature, max_tokens, df):
    try:
        llm = ChatGoogleGenerativeAI(google_api_key=api_key, model=model, temperature=temperature, max_output_tokens=max_tokens)
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
        

def main():
    st.title("AIDA: AI Data Analyzer🤖🧑‍💻")
    st.caption("Effortlessly analyze and visualize your data with AI-powered insights using Google's Gemini.")

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
        }
        .icon-container {
            display: flex;
            justify-content: space-around;
        }
        .icon-container a img {
            width: 32px;
            height: 32px;
        }
        </style>
        <br><div class="center-text"> Made with ❤️ by Akshat Gupta</div><br>
        <div class="icon-container">
            <a href="https://github.com/akshat122402" target="_blank">
                <img src="https://img.icons8.com/material-outlined/48/000000/github.png" alt="GitHub"/>
            </a>
            <a href="https://www.kaggle.com/akshatgupta7" target="_blank">
                <img src="https://img.icons8.com/windows/32/kaggle.png" alt="GitHub"/>
            </a>
            <a href="https://www.linkedin.com/in/akshat-gupta-a82923227/" target="_blank">
                <img src="https://img.icons8.com/material-outlined/48/000000/linkedin.png" alt="LinkedIn"/>
            </a>
            <a href="mailto:akki2429@gmail.com" target="_blank">
                <img src="https://img.icons8.com/ios-filled/50/gmail-new.png" alt="Gmail"/>
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
                st.dataframe(df)

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

                with st.form(key='user_input_form'):
                    user_input = st.text_input("You:", key="user_input")
                    submit_button = st.form_submit_button(label='Submit')

                if submit_button and user_input:
                    response = eda_agent.invoke(user_input)
                    output = response['output']

                    code_pattern = r'```python\n(.*?)```'
                    match = re.search(code_pattern, output, re.DOTALL)

                    if match:
                        code = match.group(1)
                        if "plt" in code or "matplotlib" in code:  
                            try:
                                exec(code, globals(), locals())

                                buf = BytesIO()
                                plt.savefig(buf, format='png')
                                buf.seek(0)

                                img_str = base64.b64encode(buf.read()).decode()
                                st.session_state.chat_history.append({"user": user_input, "ai": img_str, "type": "plot"})
                                
                                st.image(buf)
                                buf.close()

                                plt.clf()
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
