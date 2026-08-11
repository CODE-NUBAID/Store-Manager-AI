# 🏪 Store Manager AI

An AI-powered retail intelligence dashboard that combines **XGBoost sales forecasting** with a **Gemini-powered conversational AI agent**.

Store managers can ask questions about sales, view recent performance, check model accuracy, and forecast the next day's revenue through a simple web dashboard.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=flat&logo=flask)
![XGBoost](https://img.shields.io/badge/XGBoost-Forecasting-orange?style=flat)
![Gemini](https://img.shields.io/badge/Gemini-AI_Agent-4285F4?style=flat&logo=google)
![LangChain](https://img.shields.io/badge/LangChain-ReAct-1C3A57?style=flat)

---

## ✨ Features

- 💬 **Conversational AI** — Ask questions about store performance in plain English.
- 📈 **Sales Forecasting** — XGBoost predicts next-day revenue.
- 📊 **Interactive Dashboard** — View sales, customers, promotions, and recent trends.
- 🤖 **AI Agent** — LangChain ReAct agent powered by Gemini selects the appropriate tool.
- 📉 **Inline Charts** — Sales and trend questions can generate charts directly in the chat.
- ⟳ **Model Retraining** — Retrain the forecasting model from the dashboard.
- 🚀 **Quick Questions** — Predefined queries for common store-manager tasks.

---

## 🏗️ Architecture

```text
              ┌──────────────────────┐
              │     Web Dashboard    │
              │    HTML · CSS · JS   │
              └──────────┬───────────┘
                         │
                    HTTP / JSON
                         │
              ┌──────────▼───────────┐
              │      Flask API       │
              │       app.py         │
              └───────┬───────┬──────┘
                      │       │
          ┌───────────▼─┐   ┌─▼────────────────┐
          │  AI Agent   │   │    ML Model      │
          │  LangChain  │   │     XGBoost      │
          │   Gemini    │   │   Forecasting    │
          └─────────────┘   └──────────────────┘
```

### AI Agent Tools

| Tool | Purpose |
|---|---|
| `get_recent_sales_data` | Retrieves recent sales and store activity |
| `get_store_metadata` | Retrieves store and competition information |
| `forecast_next_day_sales` | Predicts next-day revenue |
| `get_model_accuracy` | Returns RMSE, MAE, and MAPE |

---

## 📁 Project Structure

```text
store_manager_ai/
├── app.py
├── requirements.txt
├── .env.example
│
├── src/
│   ├── agent_utils.py
│   ├── model_utils.py
│   ├── salesmodel.pkl
│   ├── metrics.pkl
│   └── data/
│       ├── train.csv
│       ├── store.csv
│       └── recentdata.csv
│
├── templates/
│   └── index.html
│
└── static/
    ├── css/
    │   └── style.css
    └── js/
        ├── events.js
        ├── chat.js
        ├── api.js
        ├── ui.js
        └── utils.js
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Google AI Studio API key
- Rossmann Store Sales dataset

### 1. Clone the repository

```bash
git clone https://github.com/your-username/store-manager-ai.git
cd store-manager-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the API key

Copy the environment file:

```bash
cp .env.example .env
```

Then add your API key:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

### 4. Add the dataset

Place the Rossmann files inside:

```text
src/data/
├── train.csv
└── store.csv
```

### 5. Train the model

```bash
python -c "from src.model_utils import train_and_save_model; train_and_save_model()"
```

This creates the trained model, metrics, and recent sales data.

### 6. Start the application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## 💡 Example Queries

Try questions such as:

```text
Forecast sales for tomorrow
```

```text
Show me sales trends for the last 7 days
```

```text
What is the store type?
```

```text
What is the model accuracy?
```

```text
How did sales change over the past week?
```

---

## 🧠 Machine Learning

The forecasting model uses **XGBoost** and is trained on historical Rossmann sales data.

### Features

- Calendar features
- Day-of-week and month cyclic encoding
- Promotions
- State and school holidays
- Previous-day sales
- Previous-week sales
- 7-day rolling average
- 30-day rolling average

The project uses a **time-based train/test split** to avoid data leakage.

Lag and rolling features are generated using previous observations only.

### Model Configuration

```python
XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

### Benchmark

| Metric | Value |
|---|---:|
| RMSE | 416.13 |
| MAE | 333.97 |
| MAPE | **7.72%** |

---

## ⚙️ API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Dashboard |
| `/ask` | POST | Send query to AI agent |
| `/chart-data` | GET | Retrieve recent sales data |
| `/metrics` | GET | Retrieve model metrics |
| `/train` | POST | Retrain the model |

Example `/ask` request:

```json
{
  "query": "Forecast sales for tomorrow"
}
```

---

## 📦 Tech Stack

- **Python**
- **Flask**
- **XGBoost**
- **Pandas / NumPy**
- **Scikit-learn**
- **LangChain**
- **Google Gemini**
- **Chart.js**
- **HTML / CSS / JavaScript**

---

## 🔧 Customization

### Change the Target Store

In `model_utils.py`, change:

```python
store_id=1
```

to the desired store ID.

### Change Chart Triggers

In `app.py`, modify:

```python
CHART_TRIGGERS = [
    "trend",
    "chart",
    "graph",
    "plot",
    "sales",
    "week",
    "day",
    "history"
]
```

### Change the Gemini Model

The LLM configuration can be changed in `agent_utils.py` to another model supported by `langchain-google-genai`.

---

## 📜 License

MIT License — free to use, modify, and distribute.

---

## 🎯 Project Goal

**Store Manager AI** bridges traditional machine learning and generative AI to make retail analytics easier to understand.

Instead of manually analyzing sales data, managers can simply ask questions like:

> **"Forecast sales for tomorrow."**

and receive an AI-generated answer based on the underlying XGBoost forecasting model and store data.
