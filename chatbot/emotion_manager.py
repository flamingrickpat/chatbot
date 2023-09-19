from transformers import pipeline

from chatbot.global_state import GlobalState
class EmotionManger:
    def __init__(self):
        self.emotion_classifier = None
        self.nsfw_classifier = None
        self.gs = GlobalState()

        self.init_emotion_manger()

    def init_emotion_manger(self):
        self.nsfw_classifier = pipeline("sentiment-analysis", model=self.gs.config["nsfw_classifier"])
        self.emotion_classifier = pipeline("text-classification", model=self.gs.config["emotion_classifier"])

    def nsfw_ratio(self, text: str) -> float:
        try:
            label = self.nsfw_classifier(text)[0]["label"]
            score = self.nsfw_classifier(text)[0]["score"]
            if label == "SFW":
                return 1 - score
            return score
        except Exception as e:
            return 0.5

    def get_emotions(self, text: str):
        return self.emotion_classifier(text, truncation=False)
    


