import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

app = Flask(__name__)
agent_executor = None

def get_agent():
    global agent_executor
    if agent_executor is None:
        from src.agent_utils import init_agent
        agent_executor = init_agent()
    return agent_executor

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "No query provided"}), 400
    try:
        agent = get_agent()
        response = agent.invoke({"messages": [{"role": "user", "content": query}]})
        
        # Extract the text from the response
        if isinstance(response, dict) and "messages" in response:
            messages = response["messages"]
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, 'content'):
                    content = last_msg.content
                    if isinstance(content, list) and content:
                        text_part = content[0]
                        if isinstance(text_part, dict) and 'text' in text_part:
                            output = text_part['text']
                        else:
                            output = str(content)
                    else:
                        output = str(content)
                else:
                    output = str(last_msg)
            else:
                output = "No response"
        else:
            output = str(response)
        
        return jsonify({"response": output})
    except Exception as e:
        err = str(e)
        if "contents are required" in err or "400" in err:
            return jsonify({"error": "Gemini rejected the request — try rephrasing your question."}), 500
        return jsonify({"error": err}), 500

@app.route("/train", methods=["POST"])
def train():
    try:
        from src.model_utils import train_and_save_model
        train_and_save_model()
        global agent_executor
        agent_executor = None
        return jsonify({"message": "Model trained successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)