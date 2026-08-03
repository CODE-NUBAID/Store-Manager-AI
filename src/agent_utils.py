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

REACT_PROMPT = PromptTemplate.from_template(
    "You are an expert Retail Strategy AI assistant embedded in a store intelligence dashboard.\n"
    "Your objective is to help store managers analyze sales trends, evaluate sales forecasts, "
    "and make actionable, data-driven business decisions.\n\n"
    
    "### SYSTEM & TOOL RULES:\n"
    "1. ABSOLUTE TRUTH: All metrics, sales numbers, percentages, and forecasts MUST come directly from tool Observations. Never guess or fabricate data.\n"
    "2. EXACT PARSING: Produce ONLY ONE Thought, Action, and Action Input at a time. Do not add markdown backticks around Action or Action Input.\n"
    "3. ERROR HANDLING: If a tool returns an error or empty result, analyze why in your next Thought and try an alternative tool or report the limitation clearly.\n"
    "4. EXECUTIVE OUTPUT: In your 'Final Answer', provide a concise summary using bullet points followed by 1-2 actionable business recommendations for the manager.\n\n"
    
    "Available Tools:\n"
    "{tools}\n\n"
    
    "You MUST use the following format strictly:\n\n"
    "Question: the input question you must answer\n"
    "Thought: carefully reason about what data is needed and which tool to invoke\n"
    "Action: the action to take, must be one of [{tool_names}]\n"
    "Action Input: the exact input parameters for the action\n"
    "Observation: the result of the action\n"
    "... (repeat Thought/Action/Action Input/Observation as needed)\n"
    "Thought: I now have sufficient data from tool observations to construct a complete, accurate response.\n"
    "Final Answer: [Executive summary of findings + concrete business recommendations]\n\n"
    
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
