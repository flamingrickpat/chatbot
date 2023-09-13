import os
from typing import Dict, List, Any, Tuple
import time
import logging

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizerFast, StoppingCriteriaList

from chatbot.global_state import GlobalState
from chatbot.exceptions import *
from chatbot.model import ModelApi, ModelHf

logger = logging.getLogger('model_manager')

class ModelManager():
    def __init__(self):
        self.tokenizer = None
        self.model = None

        self.telegram_chat_id = 0
        self.telegram_message_id = 0
        self.telegram_context = None

        self.gs = GlobalState()

        self.init_tokenizer()
        self.init_model()

    def init_tokenizer(self) -> None:
        """Initialize tokenizer so length of messages can be calculated."""
        tokenizer_path = self.gs.config["tokenizer_path"]
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
            device_map="cpu"
        )
        self.tokenizer = tokenizer

    def get_token_count(self, prompt: str) -> int:
        """Tokenize prompt and get length + 1 (just to be safe)"""
        inputs = self.tokenizer.encode(
            prompt,
            return_tensors="pt"
        )
        return inputs.shape[1] + 1

    def init_model(self) -> None:
        """Initialize model."""
        model_type = self.gs.config["model"]
        if model_type == "hf":
            self.model = ModelHf()
        elif model_type == "api":
            self.model = ModelApi()

    def get_message(self, prompt: str, stop_words: [str]) -> str:
        result = self.model.get_response(prompt, 250, stop_words)
        return result


