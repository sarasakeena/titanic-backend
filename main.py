from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64, os

from dotenv import load_dotenv
load_dotenv()

import requests
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.language_models.llms import LLM
from typing import Optional, List, Any

app = FastAPI()
df = pd.read_csv("titanic.csv")

# --------- Together API LLM Wrapper (Stable) ----------
class TogetherLLM(LLM):
    model: str = "mistralai/Mistral-7B-Instruct-v0.1"
    api_key: str = os.getenv("TOGETHER_API_KEY", "")

    @property
    def _llm_type(self) -> str:
        return "together_http"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        url = "https://api.together.xyz/v1/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0
        }
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["text"]
        return text.strip()

llm = TogetherLLM()

agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=False,
    allow_dangerous_code=True
)

class Question(BaseModel):
    question: str

def fig_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

@app.get("/health")
def health():
    return {"key_loaded": bool(os.getenv("TOGETHER_API_KEY"))}

@app.post("/ask")
def ask_question(q: Question):
    question = q.question.lower()

    try:
        # --- Visualizations required by assignment ---
        if "histogram" in question and "age" in question:
            plt.figure()
            df["Age"].dropna().hist(bins=20)
            plt.title("Histogram of Passenger Ages")
            return {"answer": "Here is the histogram of passenger ages.", "plot": fig_to_base64()}

        if "average" in question and "fare" in question:
            avg = df["Fare"].mean()
            plt.figure()
            df["Fare"].plot(kind="box")
            plt.title("Fare Distribution")
            return {"answer": f"The average ticket fare was {avg:.2f}.", "plot": fig_to_base64()}

        if "embark" in question:
            counts = df["Embarked"].value_counts()
            plt.figure()
            counts.plot(kind="bar")
            plt.title("Passengers by Embarkation Port")
            return {"answer": f"Passengers by port: {counts.to_dict()}", "plot": fig_to_base64()}

        if "percentage" in question and "male" in question:
            pct = (df["Sex"] == "male").mean() * 100
            plt.figure()
            df["Sex"].value_counts().plot(kind="bar")
            plt.title("Male vs Female Passengers")
            return {"answer": f"{pct:.2f}% of passengers were male.", "plot": fig_to_base64()}

        # --- Otherwise: LangChain agent for general questions ---
        response = agent.run(q.question)
        return {"answer": response}

    except Exception as e:
        return {"answer": str(e)}