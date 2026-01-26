# 🛒 Store Manager AI Agent – Sales Forecasting

## 🚀 Overview
This project implements a **tool-augmented AI agent** that combines:

- A deep learning **LSTM sales forecasting model**
- A **Large Language Model (LLM)** with reasoning and tool-calling
- A **natural-language interface** for retail decision support

The system allows business users to interact with real sales data and forecasts using plain English while relying on real computation, not hallucinated responses.

## 🧠 What Makes This an AI Agent?
This system goes beyond a traditional chatbot because:

- The agent **decides when to call tools**
- It **executes real machine learning models**
- It **interacts with external data sources**
- It follows a **reason → act → observe** loop
- Outputs are grounded in real data and predictions

## 📊 Features
- 📈 **Next-day sales forecasting**
- 🕒 **Historical sales inspection**
- 🏬 **Store metadata analysis**
- 💬 **Natural-language querying**

## 🧩 Architecture
User → LLM Agent → Tool Selection → LSTM Forecast Model → Historical Data → Response

## 📦 Tech Stack
- **Python**
- **TensorFlow / Keras**
- **LangChain**
- **Google Gemini**
- **Kaggle Datasets**

## 🏆 Key Learning
This project was built as part of the **Google × Kaggle 5-Day AI Agent Course**, demonstrating how traditional ML models can be **orchestrated by LLM-based agents** to build real-world, decision-support systems.
