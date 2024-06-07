# AIDA: AI Data Analyzer 🤖🧑‍💻

Welcome to AIDA, the AI-powered data analyzer that effortlessly helps you analyze and visualize your data using Google's Gemini model. This application leverages the power of AI to bring insightful analysis and stunning visualizations to your dataset. 

## Features

- **Easy Data Upload:** Support for CSV, Excel, and DB files.
- **AI-Powered Analysis:** Leverage Google's Gemini model to generate insights from your data.
- **Interactive Visualization:** Automatically generate and display data visualizations.
- **Streamlit Integration:** Fully interactive application using the Streamlit framework.
- **Persistent Session History:** Keep track of your interactions with the AI model and review them anytime.

## Quick Start

1. **Clone the Repository**
    ```sh
    git clone https://github.com/your-github-username/ai-data-analyzer.git
    cd ai-data-analyzer
    ```

2. **Set Up Python Environment**
    Ensure you have Python 3.12.0 installed.

    Optionally, create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies**
    ```sh
    pip install -r requirements.txt
    ```

4. **Run the Application**
    ```sh
    streamlit run app.py
    ```

5. **Visit the Application**
    Open your browser and go to [http://localhost:8501](http://localhost:8501)

## Usage

### Configuration
- **Select Model:** Choose from models like `gemini-1.5-pro`, `gemini-1.5-flash`, or `gemini-1.0-pro`.
- **Enter API Key:** Input your Google API key for accessing Gemini.
- **Adjust AI Parameters:** Use the sliders to set Temperature and Max Tokens for controlling the AI's responses.

### Upload Dataset
Upload your dataset in CSV, Excel, or DB file formats. Preview the data and proceed with interacting with the AI to gain insights and create visualizations.

### Chat Interface
Engage with the AI model through chat. Ask questions about your data and receive textual or visual insights.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

### Links
- [Live Demo](https://ai-data-analyzer.streamlit.app/) 

Please feel free to open issues or pull requests on our [GitHub repository](https://github.com/akshat122402/AIDA).

Happy Analyzing! 🚀
