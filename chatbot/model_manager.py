import os
from typing import Dict, List, Any, Tuple
import time
import logging

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizerFast, StoppingCriteriaList

from chatbot.global_state import GlobalState
from chatbot.exceptions import *
from chatbot.model_utils import StoppingCriteriaSub

logger = logging.getLogger('model_manager')

class ModelManager():
    def __init__(self):
        self.tokenizer = None
        self.model = None

        self.telegram_chat_id = 0
        self.telegram_message_id = 0
        self.telegram_context = None

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
            device_map=self.device,
            early_stopping=True
        )

    def get_token_count(self, prompt: str) -> int:
        """Tokenize prompt and get length + 1 (just to be safe)"""
        inputs = self.tokenizer.encode(
            prompt,
            return_tensors="pt"
        )
        return inputs.shape[1] + 1

    def get_message(self, prompt: str, stop_words: [str]) -> str:
        """
        Get output from prompt.
        :param prompt:
        :param stop_word:
        :return:
        """
        tokenized = self.tokenizer(prompt, return_tensors="pt").to('cuda:0')

        stopping_criteria_list = StoppingCriteriaList([StoppingCriteriaSub(stop_strings=stop_words,
                                                                           prompt_length=tokenized.input_ids.shape[1],
                                                                           tokenizer=self.tokenizer)])

        token = self.model.generate(**tokenized,
                                    max_new_tokens=250,
                                    do_sample=True,
                                    temperature=0.9,
                                    repetition_penalty=1.18,
                                    #eos_token_id=[],
                                    stopping_criteria=stopping_criteria_list,
                                    early_stopping=True)

        output = self.tokenizer.decode(token[0][tokenized.input_ids.shape[1]:])
        output = output.strip()

        tmp = output.encode('ascii', 'ignore').decode('ascii')
        logger.info(f"New output: {tmp}")

        # Remove emojis
        output = output.encode('ascii', 'ignore').decode('ascii').strip()

        return output
