import os
from typing import Dict, List, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizerFast

from chatbot.global_state import GlobalState
from chatbot.exceptions import *

class ModelManager():
    def __init__(self):
        self.tokenizer = None
        self.gs = GlobalState()

        self.init_tokenizer()

    def init_tokenizer(self):
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





