import re

from chatbot.summary.summary_base import SummaryBase
from chatbot.global_state import GlobalState

from chatbot.query_templates import summary_template

class SummaryModel(SummaryBase):
    def init_summarizer(self) -> None:
        gs = GlobalState()
        self.gs = gs

    def summarize_text(self, text: str) -> str:
        query = summary_template.replace("<conv>", text)
        summary = self.gs.model_manager.get_message(query, stop_words=["</s>"])
        summary = re.sub('[^a-zA-Z,.!? ]+', '', summary)
        summary = summary.replace("\n", " ")
        return summary
