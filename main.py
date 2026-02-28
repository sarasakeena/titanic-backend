from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64, os

from dotenv import load_dotenv

load_dotenv()

from mistralai import Mistral

from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.language_models.llms import LLM
from typing import Optional, List, Any

app = FastAPI()

# Load dataset
df = pd.read_csv("titanic.csv")


# --------- Mistral HTTP LLM Wrapper ----------
class MistralLLM(LLM):
    model: str = "mistral-small-latest"
    api_key: str = os.getenv("MISTRAL_API_KEY", "")

    @property
    def _llm_type(self) -> str:
        return "mistral_http"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY is missing")

        client = Mistral(api_key=self.api_key)

        response = client.chat.complete(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.choices[0].message.content.strip()


# LangChain Pandas Agent
llm = MistralLLM()

agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=False,
    allow_dangerous_code=True,
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
    return {"key_loaded": bool(os.getenv("MISTRAL_API_KEY"))}


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
        result = agent.invoke({"input": q.question})
        return {"answer": result["output"]}

    except Exception as e:
        return {"answer": str(e)}