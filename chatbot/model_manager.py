import os
from typing import Dict, List, Any, Tuple
import time

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizerFast

from chatbot.global_state import GlobalState
from chatbot.exceptions import *

class ModelManager():
    def __init__(self):
        self.tokenizer = None
        self.model = None

        self.gs = GlobalState()
        self.device = self.gs.config["device"]

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

    def init_model(self) -> None:
        """Initialize model."""
        model_path = self.gs.config["model_path"]
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            device_map=self.device
        )

    def get_token_count(self, prompt: str) -> int:
        """Tokenize prompt and get length + 1 (just to be safe)"""
        inputs = self.tokenizer.encode(
            prompt,
            return_tensors="pt"
        )
        return inputs.shape[1] + 1

    def get_message(self, prompt: str, stop_word: str) -> str:
        """
        Get output from prompt.
        :param prompt:
        :param stop_word:
        :return:
        """
        inputs = self.tokenizer.encode(
            prompt,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model.generate(
            inputs,
            max_new_tokens=256,
        )

        output_str = self.tokenizer.decode(outputs[0])
        return output_str




