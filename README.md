# 🧠 AI-Powered Data Insights App

This app provides an interactive, intelligent interface to explore, preprocess, visualize, and analyze datasets using powerful large language models (LLMs) from NVIDIA and a suite of data science tools. Built with **Streamlit**, it enables users to interact with their data using natural language queries.

## 🚀 Features

- **Natural Language Interface**: Ask questions or give commands like "Show correlation heatmap" or "Impute missing values with mean".
- **Preprocessing Automation**: Automatically handle missing values, outliers, encoding, scaling, datetime parsing, and more.
- **Visualization Engine**: Generates interactive charts using Plotly, Seaborn, and Matplotlib based on user queries.
- **LLM-Driven Suggestions**: Get preprocessing steps, ML model recommendations, and visualizations powered by NVIDIA NIM models.
- **Caching System**: Smart SQLite-based cache for queries, preprocessing results, and visualizations with TTL and hashing.
- **Custom Preprocessing**: Supports natural language custom commands like "log transform column 'price'" or "drop column 'ID'".
- **Data Optimization**: Automatic sampling/aggregation of large datasets for efficient visualization.

## 🧰 Tech Stack

- **Python**, **Streamlit**
- **Pandas**, **NumPy**, **Scikit-learn**, **Seaborn**, **Matplotlib**, **Plotly**
- **NVIDIA NIM API** (e.g., `nvidia/llama-3.1-nemotron-ultra-253b-v1`)
- **Pydantic** for data validation
- **SQLite** for caching
- **dotenv** for configuration

## 🔧 Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/ai-data-insights-app.git
   cd ai-data-insights-app
   ```
2.Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3.Set up environment variables

Create a .env file and add your NVIDIA API key:
   ```bash
   NVIDIA_API_KEY=your_actual_nvidia_api_key
   ```

4.Run the app
      ```bash
      streamlit run new.py
      ```

✨ Example Use Cases
“Impute column ‘age’ with median”

“Show interactive scatter plot of income vs spending score using Plotly”

“Extract year and month from ‘purchase_date’”

“Suggest ML models for this dataset”

⚠️ Notes
Make sure the NVIDIA_API_KEY is valid and you have access to the relevant NIM endpoints.

For large datasets, visualizations are auto-optimized for performance.

Streamlit’s session state is used for managing user preferences and optimization toggles.
