from transformers import pipeline

from chatbot.summary.summary_base import SummaryBase
from chatbot.global_state import GlobalState

class SummaryBart(SummaryBase):
    def init_summarizer(self) -> None:
        gs = GlobalState()
        model = gs.config["bart_summarizer"]
        self.summarizer = pipeline("summarization", model=model)

    def summarize_text(self, text: str) -> str:
        return self.summarizer(text)[0]["summary_text"]
