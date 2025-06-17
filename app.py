from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage

app = Flask(__name__)
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Modelo e agente com memória persistente entre requisições (modo simples)
system_prompt = (
    "Você é Satoru Gojo, o feiticeiro mais forte de Jujutsu Kaisen. Responda com humor, confiança, "
    "e ensine sobre maldições, energia amaldiçoada, técnicas e personagens com base no anime."
)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
memory.chat_memory.add_message(SystemMessage(content=system_prompt))

agent = initialize_agent(
    llm=llm,
    tools=[],
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=False,
)

# Rota principal
@app.route("/", methods=["GET", "POST"])
def index():
    resposta = ""
    historico = []

    for msg in memory.chat_memory.messages[1:]:  # Ignora o system prompt
        if hasattr(msg, "content"):
            role = getattr(msg, "type", "Gojo")
            autor = "Você" if role == "human" else "Gojo"
            historico.append((autor, msg.content))

    if request.method == "POST":
        pergunta = request.form["pergunta"]
        try:
            resposta = agent.run(pergunta)
            historico.append(("Você", pergunta))
            historico.append(("Gojo", resposta))
        except Exception as e:
            resposta = f"Erro: {e}"
            historico.append(("Erro", resposta))

    return render_template("chat.html", historico=historico[-20:])

if __name__ == "__main__":
    app.run(debug=True)
