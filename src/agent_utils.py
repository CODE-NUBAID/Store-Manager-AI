import pickle
import pandas as pd
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

from src.model_utils import add_features, FEATURE_COLUMNS

loaded_model = None
df_recent    = None
metrics      = None


@tool
def get_recent_sales_data(days: int = 7):
    """Returns the most recent sales data for the given number of days."""
    return df_recent.tail(days)[['Date', 'Sales', 'Customers', 'Open', 'Promo']].to_string()


@tool
def get_store_metadata():
    """Returns store metadata like type, assortment, and competition distance."""
    meta = df_recent.iloc[0][['StoreType', 'Assortment', 'CompetitionDistance']]
    return meta.to_string()


@tool
def forecast_next_day_sales():
    """Forecasts next day sales using the trained XGBoost model."""
    recent = df_recent.copy()
    recent['Date'] = pd.to_datetime(recent['Date'])
    recent = recent.sort_values('Date')

    featured = add_features(recent[recent['Open'] == 1])
    last_row = featured.dropna(subset=FEATURE_COLUMNS).iloc[[-1]]
    pred = float(loaded_model.predict(last_row[FEATURE_COLUMNS])[0])

    accuracy = f" (model MAPE: {metrics['mape']}%)" if metrics else ""
    return f"Forecasted Sales: ${pred:.2f}{accuracy}"


@tool
def get_model_accuracy():
    """Returns model evaluation metrics: RMSE, MAE, MAPE."""
    if not metrics:
        return "No metrics available — train the model first."
    return f"RMSE: {metrics['rmse']} | MAE: {metrics['mae']} | MAPE: {metrics['mape']}%"


tools = [get_recent_sales_data, get_store_metadata, forecast_next_day_sales, get_model_accuracy]

REACT_PROMPT = PromptTemplate.from_template(
    "You are a Retail Strategy AI. Help the store manager with sales trends, "
    "forecasts, and business decisions.\n\n"
    "Tools:\n{tools}\n\n"
    "Format strictly:\n"
    "Question: {input}\n"
    "Thought: think step by step\n"
    "Action: one of [{tool_names}]\n"
    "Action Input: the input\n"
    "Observation: result\n"
    "... (repeat as needed)\n"
    "Thought: I now know the final answer\n"
    "Final Answer: concise answer\n\n"
    "Begin!\n\nQuestion: {input}\nThought: {agent_scratchpad}"
)


def init_agent():
    global loaded_model, df_recent, metrics

    with open('src/salesmodel.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    df_recent = pd.read_csv('src/data/recentdata.csv')
    try:
        with open('src/metrics.pkl', 'rb') as f:
            metrics = pickle.load(f)
    except FileNotFoundError:
        metrics = None

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0,
        convert_system_message_to_human=True
    )
    agent = create_react_agent(llm, tools, REACT_PROMPT)
    return AgentExecutor(agent=agent, tools=tools, verbose=True,
                         handle_parsing_errors=True, max_iterations=5)
