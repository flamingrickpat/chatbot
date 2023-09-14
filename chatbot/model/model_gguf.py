import time

import requests
import subprocess

from chatbot.global_state import GlobalState
from chatbot.model.model_base import ModelBase


class ModelGguf(ModelBase):
    def __init__(self):
        super().__init__()

    def init_model(self):
        return

    def get_response(self, prompt: str, max_token_length: int, stop_words: [str]) -> str:
        pass
