import time

import requests
import subprocess

from chatbot.global_state import GlobalState
from chatbot.model.model_base import ModelBase


class ModelApi(ModelBase):
    def __init__(self):
        super().__init__()

    def init_model(self):
        child_process = subprocess.Popen(self.gs.config["api_exe"])
        time.sleep(180)
        return

    def get_response(self, prompt: str, max_token_length: int, stop_words: [str]) -> str:
        t = {
            "n": 1,
            "max_context_length": 4096,
            "max_length": max_token_length,
            "rep_pen": 1.08,
            "temperature": 0.8,
            "top_p": 0.92,
            "top_k": 0,
            "top_a": 0,
            "typical": 1,
            "tfs": 1,
            "rep_pen_range": 256,
            "rep_pen_slope": 0.7,
            "sampler_order": [
                6,
                0,
                1,
                2,
                3,
                4,
                5
            ],
            "prompt": prompt,
            "quiet": True,
            "stop_sequence": stop_words
        }
        r = requests.post('http://localhost:5001/api/v1/generate/', json=t)
        j = r.json()

        return j["results"][0]["text"]
