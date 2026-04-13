import pickle
import pandas as pd
import tensorflow as tf
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.prompts import PromptTemplate

loaded_model = None
loaded_scaler = None
df_recent = None

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
    """Forecasts the next day's sales using the trained LSTM model."""
    last_60 = df_recent['Sales'].values[-60:].astype(float)
    last_60_scaled = loaded_scaler.transform(last_60.reshape(-1, 1))
    X_test = last_60_scaled.reshape(1, 60, 1)
    pred_scaled = loaded_model.predict(X_test, verbose=0)
    pred_inverse = loaded_scaler.inverse_transform(pred_scaled)
    return f"Forecasted Sales: ${float(pred_inverse[0][0]):.2f}"

tools = [get_recent_sales_data, get_store_metadata, forecast_next_day_sales]

# NOTE: The trailing space after "Thought:" is intentional.
# Gemini requires every message to have non-empty content.
# Without it, an empty agent_scratchpad causes "contents are required" error.
REACT_PROMPT = PromptTemplate.from_template(
    "You are a Retail Strategy AI. Help the store manager with sales trends, forecasts, and business decisions.\n\n"
    "Tools available:\n{tools}\n\n"
    "Respond using EXACTLY this format:\n\n"
    "Question: the input question\n"
    "Thought: think about what to do\n"
    "Action: one of [{tool_names}]\n"
    "Action Input: input to the action\n"
    "Observation: result of the action\n"
    "... (repeat Thought/Action/Observation as needed)\n"
    "Thought: I now know the final answer\n"
    "Final Answer: your final answer\n\n"
    "Begin!\n\n"
    "Question: {input}\n"
    "Thought: {agent_scratchpad}"
)


def init_agent():
    global loaded_model, loaded_scaler, df_recent

    loaded_model = tf.keras.models.load_model('src/salesmodel.keras')
    with open('src/scaler.pkl', 'rb') as f:
        loaded_scaler = pickle.load(f)
    df_recent = pd.read_csv('src/data/recentdata.csv')

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        convert_system_message_to_human=True  # fixes Gemini system message errors
    )

    system_prompt = """You are a Retail Strategy AI assistant. Help the store manager understand sales trends, forecasts, and make smart business decisions.

Use the available tools to get recent sales data, store metadata, and forecast future sales. Be concise and insightful."""

    agent = create_agent(llm, tools=tools, system_prompt=system_prompt)
    return agent