from transformers import pipeline
from numpy import std
from collections import deque, Counter
from statistics import stdev
from math import exp
import re

from chatbot.global_state import GlobalState
from chatbot.query_templates import emotion_template

emotion_list = [
    "caring",
    "faithful",
    "content",
    "sentimental",
    "joyful",
    "hopeful",
    "proud",
    "guilty",
    "sad",
    "grateful",
    "afraid",
    "ashamed",
    "trusting",
    "confident",
    "prepared",
    "anxious",
    "lonely",
    "terrified",
    "devastated",
    "disgusted",
    "annoyed",
    "excited",
    "jealous",
    "anticipating",
    "furious",
    "angry",
    "impressed",
    "nostalgic",
    "surprised",
    "apprehensive",
    "disappointed",
    "embarrassed",
]


class MoodAdjuster:
    def __init__(self):
        self.CONSTANTS = {
            'EMOTION_THRESHOLD_BASE': 0.5,
            'RECENT_EMOTIONS_LIMIT': 5,
            'MOOD_DECAY_RATE': 0.1,
            'MOOD_INTENSITY_RATE': 0.2,
            'MOOD_INERTIA_BASE': 0.3,
            'USER_BEHAVIOR_ADJUSTMENT_RATE': 0.05,
            'CUMULATIVE_INTENSITY_DECAY': 0.1,
            'LONG_TERM_DECAY_RATE': 0.05
        }
        self.GLOBAL_STATE = {
            'bot_mood': 0,
            'user_mood': 0,
            'bot_emotion_history': deque(maxlen=self.CONSTANTS['RECENT_EMOTIONS_LIMIT']),
            'user_emotion_history': deque(maxlen=self.CONSTANTS['RECENT_EMOTIONS_LIMIT']),
            'user_mood_changes': Counter(),
            'cumulative_mood': {'bot': 0, 'user': 0},
            'cumulative_emotion_intensity': {'bot': 0, 'user': 0},
            'recent_emotion_counter': {'bot': Counter(), 'user': Counter()},
            'long_term_emotion_avg': {'bot': 0, 'user': 0},
            'bot_mood_history': deque(maxlen=5),
            'user_mood_history': deque(maxlen=5),
        }
        self.EMOTION_SCORES = {
            'anger': -0.9,
            'fear': -0.7,
            'sadness': -0.8,
            'neutral': 0.0,
            'joy': 0.7,
            'love': 0.9,
            'surprise': 0.5  # Assuming surprise is slightly positive.
        }

    def get_global_state(self, key, subkey=None):
        if subkey:
            return self.GLOBAL_STATE[key][subkey]
        return self.GLOBAL_STATE[key]

    def set_global_state(self, key, value, subkey=None):
        if subkey:
            self.GLOBAL_STATE[key][subkey] = value
        else:
            self.GLOBAL_STATE[key] = value

    def get_recent_emotions(self, source):
        return list(self.get_global_state(f"{source}_emotion_history"))

    def adjust_decay_rate(self, source):
        mood_changes = self.get_global_state("user_mood_changes").get(source, 0)
        decay_rate = self.CONSTANTS['MOOD_DECAY_RATE'] + mood_changes * self.CONSTANTS['USER_BEHAVIOR_ADJUSTMENT_RATE']

        # Decay the mood_changes counter to avoid long-term rigidity
        self.set_global_state("user_mood_changes", mood_changes * 0.9, source)

        # Decay the counts in recent_emotion_counter to keep it relevant to recent interactions
        recent_emotion_counts = self.get_global_state("recent_emotion_counter", source)
        for emotion in recent_emotion_counts:
            recent_emotion_counts[emotion] *= 0.9
        return decay_rate

    def adjust_mood_inertia(self, source):
        recent_emotions = self.get_recent_emotions(source)
        if not recent_emotions:
            return self.CONSTANTS['MOOD_INERTIA_BASE']
        recent_emotion_scores_with_intensity = [self.normalize_emotion_score(emotion, intensity) for emotion, intensity
                                                in recent_emotions]
        variability = stdev(recent_emotion_scores_with_intensity) if len(
            recent_emotion_scores_with_intensity) > 1 else 0
        return self.CONSTANTS['MOOD_INERTIA_BASE'] * max(0.5, 1 - variability)

    def counterbalance(self, mood):
        # Modify counterbalance to be less effective as mood reaches extreme values
        return -2 / (1 + exp(-0.5 * abs(mood))) + 1

    def calculate_weighted_average_emotion(self, recent_emotions):
        if not recent_emotions:
            return 0
        weights = [i + 1 for i in range(len(recent_emotions))]
        total_weight = sum(weights)
        weighted_avg_emotion = sum(
            self.normalize_emotion_score(emotion, intensity) * weight for (emotion, intensity), weight in
            zip(recent_emotions, weights)) / total_weight
        return weighted_avg_emotion

    def normalize_emotion_score(self, emotion, intensity):
        if emotion in self.EMOTION_SCORES:
            return self.EMOTION_SCORES[emotion] * intensity
        else:
            return 0

    def update_long_term_emotion_avg(self, emotion_score, source):
        long_term_emotion_avg = self.get_global_state("long_term_emotion_avg", source)
        long_term_emotion_avg = (1 - self.CONSTANTS['LONG_TERM_DECAY_RATE']) * long_term_emotion_avg + self.CONSTANTS[
            'LONG_TERM_DECAY_RATE'] * emotion_score
        self.set_global_state("long_term_emotion_avg", long_term_emotion_avg, source)

    def identify_dominant_emotion(self, recent_emotions_counter):
        if not recent_emotions_counter:
            return None, 0
        dominant_emotion, _ = max(recent_emotions_counter.items(), key=lambda x: x[1])
        return dominant_emotion

    def sign(self, x):
        return (x > 0) - (x < 0)

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def undo_last_mood_adjustment(self, source):
        mood_history = self.get_global_state(f"{source}_mood_history")
        if mood_history:
            last_mood = mood_history.pop()
            self.set_global_state(f"{source}_mood", last_mood)
            self.set_global_state(f"{source}_mood_history", mood_history)
        else:
            print("No previous mood to revert to.")

    def calculate_initial_values(self, prev_mood, emotion_label, intensity, source):
        normalized_current_emotion_score = self.normalize_emotion_score(emotion_label, intensity)
        decay_adjustment = -self.adjust_decay_rate(source) * prev_mood
        recent_emotions = self.get_recent_emotions(source)
        weighted_avg_emotion = self.calculate_weighted_average_emotion(recent_emotions)
        return normalized_current_emotion_score, decay_adjustment, weighted_avg_emotion

    def update_global_state(self, normalized_current_emotion_score, source, emotion_label):
        self.update_long_term_emotion_avg(normalized_current_emotion_score, source)
        recent_emotion_counter = self.get_global_state("recent_emotion_counter", source)
        recent_emotion_counter.update([emotion_label])

    def calculate_mood_adjustment(self, normalized_current_emotion_score, weighted_avg_emotion, emotion_label, source):
        delta = abs(weighted_avg_emotion - normalized_current_emotion_score)
        alpha = 0.5 + delta
        mood_adjustment = alpha * normalized_current_emotion_score + (1 - alpha) * weighted_avg_emotion

        most_common_emotion, freq = self.get_global_state("recent_emotion_counter", source).most_common(1)[0]
        if most_common_emotion == emotion_label:
            mood_adjustment *= 1 + (0.1 * freq)

        long_term_adjustment = 0.1 * self.get_global_state("long_term_emotion_avg", source)
        mood_adjustment += long_term_adjustment
        return mood_adjustment

    def apply_cumulative_adjustments(self, source, normalized_current_emotion_score):
        cumulative_emotion_intensity = self.get_global_state("cumulative_emotion_intensity", source)

        # Update cumulative emotion intensity based on the current emotion score
        cumulative_emotion_intensity += abs(normalized_current_emotion_score)

        # Apply decay if the cumulative intensity is above a threshold
        if cumulative_emotion_intensity > 10:
            cumulative_emotion_intensity *= 0.9

        self.set_global_state("cumulative_emotion_intensity", cumulative_emotion_intensity, source)

        cumulative_mood = self.get_global_state("cumulative_mood", source)
        return cumulative_mood, cumulative_emotion_intensity

    def finalize_mood_update(self, prev_mood, mood_change, decay_adjustment, source):
        new_mood = prev_mood + mood_change + decay_adjustment
        new_mood += self.counterbalance(new_mood)

        mood_history = self.get_global_state(f"{source}_mood_history")
        mood_history.append(prev_mood)  # save the previous mood before updating
        self.set_global_state(f"{source}_mood_history", mood_history)

        self.set_global_state(f"{source}_mood", new_mood)
        return max(-1, min(1, new_mood))

    def adjust_mood(self, prev_mood, emotion_label, intensity, source):
        normalized_current_emotion_score, decay_adjustment, weighted_avg_emotion = self.calculate_initial_values(
            prev_mood, emotion_label, intensity, source)

        self.update_global_state(normalized_current_emotion_score, source, emotion_label)

        mood_adjustment = self.calculate_mood_adjustment(normalized_current_emotion_score, weighted_avg_emotion,
                                                         emotion_label, source)

        cumulative_mood, cumulative_emotion_intensity = self.apply_cumulative_adjustments(source,
                                                                                          normalized_current_emotion_score)

        cumulative_mood += mood_adjustment

        cumulative_emotion_intensity *= (1 - self.CONSTANTS['CUMULATIVE_INTENSITY_DECAY'])
        cumulative_emotion_intensity += abs(normalized_current_emotion_score)
        self.set_global_state("cumulative_emotion_intensity", cumulative_emotion_intensity, source)

        dynamic_inertia = self.adjust_mood_inertia(source)
        current_threshold = self.CONSTANTS['EMOTION_THRESHOLD_BASE'] * max(0, 1 - cumulative_emotion_intensity)
        mood_change = cumulative_mood if abs(cumulative_mood) >= current_threshold else 0
        mood_change *= (1 - dynamic_inertia)
        mood_change = self.sign(mood_change) * self.sigmoid(abs(mood_change))
        max_mood_change = 0.2  # Limit the maximum mood change in a single step
        mood_change = min(max_mood_change, max(-max_mood_change, mood_change))

        if emotion_label == 'neutral':
            decay_adjustment = -0.2 * self.sign(prev_mood)  # Push towards zero

        return self.finalize_mood_update(prev_mood, mood_change, decay_adjustment, source)


mood_adjuster = MoodAdjuster()

class EmotionManger:
    def __init__(self):
        self.emotion_classifier = None
        self.nsfw_classifier = None
        self.gs = GlobalState()

        self.init_emotion_manger()

        self.con = self.gs.db_manager.con
        self.cur = self.gs.db_manager.cur

    def init_emotion_manger(self):
        self.nsfw_classifier = pipeline("sentiment-analysis", model=self.gs.config["nsfw_classifier"])
        self.emotion_classifier = pipeline("text-classification", model=self.gs.config["emotion_classifier"],
                                           top_k=None)

    def nsfw_ratio(self, text: str) -> float:
        try:
            label = self.nsfw_classifier(text)[0]["label"]
            score = self.nsfw_classifier(text)[0]["score"]
            if label == "SFW":
                return 1 - score
            return score
        except Exception as e:
            return 0.5

    def get_emotions_old(self, text: str) -> list:
        # [{'label': 'anger', 'score': 0.9796756505966187}, {'label': 'sadness', 'score': 0.010976619087159634}, {'label': 'joy', 'score': 0.0030405886936932802}, {'label': 'love', 'score': 0.002827202435582876}, {'label': 'fear', 'score': 0.0018505036132410169}, {'label': 'surprise', 'score': 0.0016293766675516963}]
        output = self.emotion_classifier(
            text,
            truncation=False,
            max_length=self.emotion_classifier.model.config.max_position_embeddings,
        )[0]
        return sorted(output, key=lambda x: x["score"], reverse=True)

    def calc_emotions(self, id):
        """
        Recalc emotions of all messages.
        :return:
        """
        if id is None:
            res = self.cur.execute("SELECT * FROM messages")
        else:
            res = self.cur.execute(f"SELECT * FROM messages where id = {id}")
        res = res.fetchall()

        for i in range(len(res)):
            id = res[i]["id"]
            character_id = res[i]["character_id"]
            is_user = res[i]["is_user"]
            message = res[i]["message"]
            character = res[i]["character"]
            token_count = res[i]["token_count"]

            output = self.emotion_classifier(
                message,
                truncation=False,
                max_length=self.emotion_classifier.model.config.max_position_embeddings,
            )[0]
            for d in output:
                lbl = d["label"]
                sql = f"update messages set {lbl} = ? where id = ?"
                self.cur.execute(sql, (d["score"], id))

        self.con.commit()


    def get_emotion_from_ids(self, is_message, character_id, ids):
        res = self.cur.execute("select name from characters where id = ?", (character_id,)).fetchall()
        name = res[0]["name"]

        emotion_map = {}
        for emotion in emotion_list:
            emotion_map[emotion] = 0
        for i, id in enumerate(ids):
            if is_message:
                res = self.cur.execute(f"select * from messages where id = ?", (id,)).fetchall()
            else:
                res = self.cur.execute(f"select * from summaries where id in (select summary_id from graph_context where id = {id})").fetchall()
            if len(res) > 0:
                for emotion in emotion_list:
                    scaled_em = res[0][emotion]
                    if is_message:
                        scaled_em = res[0][emotion] * (i / len(ids))
                    emotion_map[emotion] = emotion_map[emotion] + scaled_em

        as_text = ""
        for key, value in emotion_map.items():
            as_text = as_text + key + ": " + "{:.2f}".format(value) + "\n"

        query = emotion_template
        query = query.replace("<emotions>", as_text.strip())
        query = query.replace("<name>", name.strip())

        summary = self.gs.model_manager.get_message(query, stop_words=["</s>"])
        summary = re.sub('[^a-zA-Z,.!? ]+', '', summary)
        summary = summary.replace("\n", " ")

        return summary
