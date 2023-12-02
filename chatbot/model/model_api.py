import time

import requests
import subprocess
import logging

from chatbot.global_state import GlobalState
from chatbot.model.model_base import ModelBase

logger = logging.getLogger('model_api')

class ModelApi(ModelBase):
    def __init__(self):
        super().__init__()

    def init_model(self):
        return

    def get_response(self, prompt: str, max_token_length: int, stop_words: [str]) -> str:
        self.gs = GlobalState()

        while True:
            try:
                t = {
                    "n": 1,
                    "max_context_length": self.gs.config["context_size"],
                    "max_length": max_token_length,
                    "rep_pen": 1.08,
                    "temperature": 0.7,
                    "top_p": 0.92,
                    "top_k": 100,
                    "top_a": 0,
                    "typical": 1,
                    "tfs": 1,
                    "rep_pen_range": 320,
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
                msg = j["results"][0]["text"]
                msg = msg.replace("</s>", "")
                msg = msg.strip()

                return msg
            except Exception as e:
                logger.error(str(e))
