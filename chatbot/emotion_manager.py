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
        self.emotion_classifier = pipeline("text-classification", model=self.gs.config["emotion_classifier"], top_k=None)

    def nsfw_ratio(self, text: str) -> float:
        try:
            label = self.nsfw_classifier(text)[0]["label"]
            score = self.nsfw_classifier(text)[0]["score"]
            if label == "SFW":
                return 1 - score
            return score
        except Exception as e:
            return 0.5

    def get_emotions(self, text: str) -> list:
        # [{'label': 'anger', 'score': 0.9796756505966187}, {'label': 'sadness', 'score': 0.010976619087159634}, {'label': 'joy', 'score': 0.0030405886936932802}, {'label': 'love', 'score': 0.002827202435582876}, {'label': 'fear', 'score': 0.0018505036132410169}, {'label': 'surprise', 'score': 0.0016293766675516963}]
        output = self.emotion_classifier(
            text,
            truncation=False,
            max_length=self.emotion_classifier.model.config.max_position_embeddings,
        )[0]
        return sorted(output, key=lambda x: x["score"], reverse=True)
    


