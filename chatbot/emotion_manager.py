from transformers import pipeline

from chatbot.global_state import GlobalState
class EmotionManger:
    def __init__(self):
        self.nsfw_classifier = None
        self.gs = GlobalState

        self.init_emotion_manger()

    def init_emotion_manger(self):
        self.nsfw_classifier = pipeline("sentiment-analysis", model=self.gs.config["nsfw_classifier"])

    def nsfw_ratio(self, text: str) -> float:
        label = self.nsfw_classifier(text)[0]["label"]
        score = self.nsfw_classifier(text)[0]["score"]
        if label == "SFW":
            return 1 - score
        return score
