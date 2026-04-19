# 🛒 Store Manager AI: Retail Intelligence Dashboard

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Gemini](https://img.shields.io/badge/Google_Gemini-2.0_Flash-4285F4?style=flat&logo=google&logoColor=white)](https://ai.google.dev/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)

An advanced, agentic retail intelligence platform that empowers store managers with **Predictive Analytics** and **AI-Driven Strategy**. This project integrates a **Long Short-Term Memory (LSTM)** neural network for sales forecasting with a **ReAct AI Agent** that can "reason" through store data to provide actionable insights.

## 🚀 Key Features

- **Brain-Power (AI Agent):** Powered by Gemini 2.5 Flash, the agent uses a ReAct logic loop to decide which tools to use (Forecasting, Metadata, or Historical Analysis) to answer complex manager queries.
- **Predictive Forecasting:** A deep learning LSTM model trained on historical retail data to predict the next day's sales with high accuracy.
- **Modern UI:** A high-performance, dark-themed dashboard built with Flask, featuring a minimalist "Terminal" aesthetic.
- **Self-Correcting Pipeline:** Built-in "Retrain Engine" allows the manager to update the underlying LSTM model directly from the UI.

## 🛠️ Tech Stack

- **Backend:** Flask (Python)
- **AI/LLM:** Google Gemini via LangChain (ReAct Agent Framework)
- **Machine Learning:** TensorFlow, Keras, Scikit-learn (LSTM Architecture)
- **Frontend:** HTML5, CSS3 (Modern UI), JavaScript
- **Environment:** Dotenv for secure API management

## 📂 Project Structure

```text
Store manager ai/
├── app.py              # Flask Server & Route Management
├── src/
│   ├── agent_utils.py  # LangChain Agent logic & Tool definitions
│   ├── model_utils.py  # LSTM Training & Inference pipeline
│   └── data/           # CSV datasets (Train, Store, Recent)
├── static/
│   ├── style.css       # Custom Cyber-Dashboard CSS
│   └── app.js          # Async API calls
└── templates/
    └── index.html      # Main Dashboard Interface
