import abc

class SummaryBase:
    def __init__(self):
        pass

    def init_summarizer(self) -> None:
        """
        Init summarizer.
        """
        raise NotImplementedError()

    def summarize_text(self, text: str) -> str:
        """
        Summarize text.
        """
        raise NotImplementedError()