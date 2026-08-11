# 🏪 Store Manager AI

A retail intelligence dashboard that combines **XGBoost sales forecasting** with a **Gemini-powered AI agent** — giving store managers a conversational interface to query sales data, predict tomorrow's revenue, and understand store performance in plain English.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=flat&logo=flask)
![XGBoost](https://img.shields.io/badge/XGBoost-forecasting-orange?style=flat)
![Gemini](https://img.shields.io/badge/Gemini_2.0_Flash-AI_Agent-4285F4?style=flat&logo=google)
![LangChain](https://img.shields.io/badge/LangChain-ReAct_Agent-1C3A57?style=flat)

---

## ✨ Features

- **💬 Conversational AI** — Ask questions in plain English, such as:
  - "What were my sales last week?"
  - "Forecast tomorrow"
- **📈 Sales Forecasting** — XGBoost model trained on historical data predicts next-day revenue.
- **📊 Live Dashboard** — Real-time stats panel with sparkline chart, latest sales, customer count, and promotion status.
- **🤖 ReAct Agent** — LangChain ReAct agent powered by Gemini 2.0 Flash reasons step-by-step and picks the right tool.
- **⟳ One-click Retraining** — Retrain the model in-browser anytime new data is available.
- **📉 Inline Charts** — When you ask about trends or sales history, a bar chart appears directly inside the chat.
- **🚀 Quick-ask Buttons** — Pre-built queries for common store-manager questions.

---

## 🏗️ Architecture

```text
┌─────────────────────────────────────────────────────────┐
│                     Browser (UI)                        │
│  sidebar · topbar · chat log · stats panel · charts     │
│  JS modules: events → chat → api → ui → utils           │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP (JSON)
┌────────────────────▼────────────────────────────────────┐
│                  Flask (app.py)                         │
│  POST /ask · GET /chart-data · GET /metrics · POST /train│
└──────────┬──────────────────────────┬───────────────────┘
           │                          │
┌──────────▼──────────┐   ┌──────────▼──────────────────┐
│   agent_utils.py    │   │      model_utils.py         │
│  LangChain ReAct    │   │  XGBoost · feature eng.     │
│  Gemini 2.0 Flash   │   │  train / predict / metrics  │
│  4 tools (below)    │   └─────────────────────────────┘
└─────────────────────┘
```

### AI Agent Tools

| Tool | Description |
|---|---|
| `get_recent_sales_data` | Returns the last N days of sales, customers, open status, and promotion information. |
| `get_store_metadata` | Returns store type, assortment type, and competition distance. |
| `forecast_next_day_sales` | Runs the XGBoost model and returns the next-day revenue prediction. |
| `get_model_accuracy` | Returns RMSE, MAE, and MAPE from the last training run. |

---

## 📁 Project Structure

```text
store_manager_ai/
├── app.py                    # Flask app — routes & chart trigger logic
├── requirements.txt
├── .env.example
│
├── src/
│   ├── agent_utils.py        # LangChain ReAct agent, Gemini LLM, 4 tools
│   ├── model_utils.py        # XGBoost training, feature engineering, metrics
│   │
│   ├── data/
│   │   ├── train.csv         # ← You provide (Rossmann format)
│   │   ├── store.csv         # ← You provide (Rossmann format)
│   │   └── recentdata.csv    # Auto-generated after training (last 90 days)
│   │
│   ├── salesmodel.pkl        # Auto-generated trained model
│   └── metrics.pkl           # Auto-generated training metrics
│
├── templates/
│   └── index.html            # Single-page dashboard shell
│
└── static/
    ├── css/
    │   └── style.css         # Inter font · CSS variables · responsive layout
    │
    └── js/
        ├── events.js         # DOM event listeners
        ├── chat.js           # sendQuery · trainModel · loadDashboard orchestration
        ├── api.js            # Fetch wrappers for all Flask endpoints
        ├── ui.js             # DOM rendering, Chart.js charts, stats updates
        └── utils.js           # escapeHtml · formatCurrency · formatPercent
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- A [Google AI Studio](https://aistudio.google.com/) API key
  - The free tier works for development.
- Rossmann Store Sales dataset:
  - `train.csv`
  - `store.csv`
  - Available on [Kaggle](https://www.kaggle.com/competitions/rossmann-store-sales/data)

---

### 1. Clone & Install

```bash
git clone https://github.com/your-username/store-manager-ai.git
cd store-manager-ai
pip install -r requirements.txt
```

---

### 2. Configure Environment

Copy the example environment file:

```bash
cp .env.example .env
```

Open `.env` and add your Google API key:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

> **Note:** Never commit your `.env` file or expose your API key publicly.

---

### 3. Add the Data Files

Place the Rossmann dataset files inside `src/data/`:

```text
src/data/
├── train.csv
└── store.csv
```

---

### 4. Train the Model

Run:

```bash
python -c "from src.model_utils import train_and_save_model; train_and_save_model()"
```

This generates:

```text
src/
├── salesmodel.pkl
├── metrics.pkl
└── data/
    └── recentdata.csv
```

---

### 5. Run the App

Start the Flask application:

```bash
python app.py
```

Then open:

**http://127.0.0.1:5000**

---

## 💡 Example Queries

Once the dashboard is running, try asking:

```text
Forecast sales for tomorrow
```

```text
Show me sales trends for the last 7 days with a chart
```

```text
What is the store type and competition distance?
```

```text
What is the model accuracy?
```

```text
How did sales change over the past week?
```

```text
Was a promotion running on the last open day?
```

You can also click the **Quick Ask** buttons in the right-side panel.

---

## 🧠 How the ML Model Works

The XGBoost model is trained on a single store (**Store 1 by default**) using engineered features.

### Feature Engineering

| Category | Features |
|---|---|
| **Calendar** | `DayOfWeek`, `DayOfMonth`, `WeekOfYear`, `Month` |
| **Cyclic Encoding** | `DOW_sin`, `DOW_cos`, `Month_sin`, `Month_cos` |
| **Events** | `Promo`, `StateHoliday`, `SchoolHoliday` |
| **Lag Features** | `Sales_lag1`, `Sales_lag7` |
| **Rolling Averages** | `Sales_roll7`, `Sales_roll30` |

### Key Design Decisions

#### Closed Days

Closed days (`Open=0`) are filtered before training.

This prevents zero-sales rows from breaking MAPE calculations and misleading the model.

#### No Data Leakage

All lag and rolling features use `.shift(1)` before rolling.

This ensures that the model never has access to future sales information while generating features.

#### Time-Based Train/Test Split

The dataset is split using an **85% / 15% time-based split**.

The data is **not shuffled**, because randomly shuffling time-series data can cause future information to leak into the training set.

#### Recent Data

The last **90 days** of raw store data, including closed days, are saved to:

```text
src/data/recentdata.csv
```

This file is used by the AI agent when answering questions about recent store performance.

---

## ⚙️ XGBoost Model Configuration

The model uses the following configuration:

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

---

## 📊 Benchmark Results

Benchmark results for **Store 1**:

| Metric | Value |
|---|---:|
| **RMSE** | 416.13 |
| **MAE** | 333.97 |
| **MAPE** | **7.72%** |

> These values are benchmark results and may vary depending on the dataset, preprocessing, feature engineering, and training configuration.

---

## 🖥️ Frontend Architecture

The JavaScript frontend is split into five modules with strict separation of concerns.

| File | Responsibility |
|---|---|
| `events.js` | The only file that attaches DOM event listeners. |
| `chat.js` | Orchestrates user actions such as sending queries, training, and loading the dashboard. |
| `api.js` | Contains all `fetch()` calls, with one function per API endpoint. |
| `ui.js` | Handles DOM rendering, Chart.js chart creation, and stats-panel updates. |
| `utils.js` | Pure helper functions with no side effects. |

### Chart.js

The UI uses **Chart.js 4.4** for:

- A sparkline in the stats panel.
- Updating the sparkline after model retraining.
- Inline bar charts injected into AI responses.
- Visualizing sales trends and historical sales data.

---

## ⚙️ API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Serves the dashboard HTML. |
| `/ask` | `POST` | Sends a query to the AI agent. Body: `{ "query": "..." }` |
| `/chart-data` | `GET` | Returns the last 7 open-day sales, customers, and promotion status. |
| `/metrics` | `GET` | Returns the latest model RMSE, MAE, and MAPE. |
| `/train` | `POST` | Triggers a full model retraining. |

### `/ask`

Example request:

```http
POST /ask
Content-Type: application/json
```

```json
{
  "query": "Forecast sales for tomorrow"
}
```

---

### `/chart-data`

Example:

```http
GET /chart-data
```

Returns recent sales information used by the dashboard charts.

---

### `/metrics`

Example:

```http
GET /metrics
```

Returns the latest model performance metrics.

---

### `/train`

Example:

```http
POST /train
```

Triggers a complete model retraining process.

---

## 🔧 Configuration & Customization

### Changing the Target Store

By default, the model trains on **Store 1**.

In `model_utils.py`, change:

```python
store_id=1
```

to any store ID present in your dataset.

---

### Adjusting Chart Trigger Keywords

The Flask application uses chart-trigger keywords to determine when an inline chart should appear alongside an AI response.

In `app.py`, edit:

```python
CHART_TRIGGERS = [
    "trend",
    "chart",
    "graph",
    "plot",
    "sales",
    "week",
    "day",
    "history",
    "show me",
    "visual"
]
```

Add or remove keywords depending on the behavior you want.

For example:

```python
CHART_TRIGGERS = [
    "trend",
    "chart",
    "graph",
    "plot",
    "sales",
    "week",
    "day",
    "history",
    "show me",
    "visual",
    "revenue"
]
```

---

### Swapping the LLM

The AI agent currently uses Gemini through `langchain-google-genai`.

In `agent_utils.py`, replace:

```text
gemini-2.0-flash-exp
```

with another model supported by `langchain-google-genai`.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `flask` | Web server and API endpoints |
| `xgboost` | Sales forecasting model |
| `scikit-learn` | RMSE / MAE metrics |
| `pandas` | Data processing |
| `numpy` | Numerical operations and feature engineering |
| `langchain` | ReAct agent framework |
| `langchain-google-genai` | Gemini LLM integration |
| `python-dotenv` | Environment variable management |

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

The application uses environment variables to keep API credentials outside the source code.

Create a `.env` file:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

A recommended `.env.example` file:

```env
GOOGLE_API_KEY=
```

Never commit the actual `.env` file to Git.

Add it to `.gitignore`:

```gitignore
.env
__pycache__/
*.pyc
*.pkl
```

---

## 🧪 Model Training Workflow

The overall training workflow is:

```text
Rossmann train.csv
        │
        ▼
Filter target store
        │
        ▼
Merge with store.csv
        │
        ▼
Sort chronologically
        │
        ▼
Remove closed days
        │
        ▼
Feature engineering
        │
        ├── Calendar features
        ├── Cyclic features
        ├── Promotion features
        ├── Holiday features
        ├── Lag features
        └── Rolling averages
        │
        ▼
Time-based 85/15 split
        │
        ▼
Train XGBoost
        │
        ▼
Evaluate model
        │
        ├── RMSE
        ├── MAE
        └── MAPE
        │
        ▼
Save model + metrics
        │
        ▼
Generate recentdata.csv
```

---

## 🤖 AI Agent Workflow

When a manager asks a question, the request follows this flow:

```text
Manager
   │
   ▼
Browser Chat UI
   │
   ▼
POST /ask
   │
   ▼
LangChain ReAct Agent
   │
   ▼
Gemini 2.0 Flash
   │
   ├── get_recent_sales_data
   │
   ├── get_store_metadata
   │
   ├── forecast_next_day_sales
   │
   └── get_model_accuracy
   │
   ▼
Tool result
   │
   ▼
Gemini generates response
   │
   ▼
Flask response
   │
   ▼
Chat UI
```

---

## 📈 Forecasting Workflow

For a forecast request such as:

```text
Forecast sales for tomorrow
```

the AI agent calls:

```text
forecast_next_day_sales
```

The tool loads the trained XGBoost model and generates the next-day revenue prediction using the required engineered features.

The result is then passed back to the Gemini agent, which converts the prediction into a natural-language response for the store manager.

---

## 📊 Data Requirements

The project expects the standard **Rossmann Store Sales** dataset.

Required files:

```text
train.csv
store.csv
```

### `train.csv`

The training dataset should contain fields such as:

- `Store`
- `DayOfWeek`
- `Date`
- `Sales`
- `Customers`
- `Open`
- `Promo`
- `StateHoliday`
- `SchoolHoliday`

### `store.csv`

The store metadata dataset contains information such as:

- `Store`
- `StoreType`
- `Assortment`
- `CompetitionDistance`
- Competition-related dates
- Promotion-related information

The exact columns required may depend on the implementation in `model_utils.py`.

---

## 🛠️ Troubleshooting

### `GOOGLE_API_KEY` not found

Make sure `.env` exists in the project root:

```text
store_manager_ai/
├── .env
├── app.py
└── ...
```

And contains:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

---

### Model file not found

If you see an error related to:

```text
salesmodel.pkl
```

run:

```bash
python -c "from src.model_utils import train_and_save_model; train_and_save_model()"
```

This creates the required model artifacts.

---

### Data files not found

Verify that the Rossmann files are located at:

```text
src/data/train.csv
src/data/store.csv
```

---

### Port already in use

If port `5000` is already being used, change the Flask port in `app.py` or stop the process currently using it.

---

## 🔮 Future Improvements

Potential improvements include:

- Multi-store forecasting.
- Longer-horizon forecasting.
- Automatic daily model retraining.
- Hyperparameter optimization.
- Additional external features such as weather and local events.
- More advanced anomaly detection.
- Store-to-store performance comparisons.
- Role-based manager access.
- Authentication and authorization.
- Persistent chat history.
- Exportable sales reports.
- More advanced interactive charts.
- Production deployment using Gunicorn or another WSGI server.
- Database integration instead of relying on CSV files.

---

## 📜 License

MIT — free to use, modify, and distribute.

---

## 👨‍💻 Project Summary

**Store Manager AI** combines traditional machine learning with generative AI to create a practical retail intelligence system.

The project brings together:

```text
Historical Sales Data
        │
        ▼
Feature Engineering
        │
        ▼
XGBoost Forecasting
        │
        ▼
Model Metrics
        │
        ▼
LangChain ReAct Agent
        │
        ▼
Gemini
        │
        ▼
Conversational Store Intelligence
        │
        ▼
Interactive Dashboard
```

Instead of requiring a store manager to manually inspect CSV files, calculate trends, or interpret ML metrics, the system provides a conversational interface where questions can be asked naturally.

For example:

> **Manager:** Forecast sales for tomorrow.

> **AI:** Tomorrow's predicted revenue is approximately ₹XX,XXX based on recent sales trends, day-of-week patterns, promotions, and other historical features.

This makes the forecasting model and underlying store data accessible through a simple conversational experience.
