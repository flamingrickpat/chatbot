import os
from typing import Dict, List, Any, Tuple
import time
import logging
import sqlite3
from datetime import datetime, timezone

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizerFast, StoppingCriteriaList


from chatbot.global_state import GlobalState
from chatbot.exceptions import *
from chatbot.model import ModelApi, ModelHf, ModelGguf
from chatbot.model import convert

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

        path = self.gs.config["database_path"]
        if os.path.exists(path):
            self.con = sqlite3.connect(path)
            self.con.row_factory = sqlite3.Row
            self.cur = self.con.cursor()

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
        elif model_type == "gguf":
            self.model = None


    def reload_model(self, character_id: int) -> None:
        """
        Reload model when switching character.
        :param character_id:
        :return:
        """

        logger.info(f"Reloading model for {character_id}...")

        model_type = self.gs.config["model"]
        if model_type == "gguf":
            path = self.get_finetuned_model_path(character_id)
            if path == "":
                logger.info(f"Creating first model")
                self.create_finetuned_model(character_id, qlora=False)
                path = self.get_finetuned_model_path(character_id)

            logger.info(f"Using model: {path}")
            self.model = ModelGguf(path)


    def get_message(self, prompt: str, stop_words: [str]) -> str:
        result = self.model.get_response(prompt, 250, stop_words)
        if self.gs.config["ascii_only"]:
            result = result.encode('ascii', 'ignore').decode('ascii')
        return result

    def get_finetuned_model_path(self, character_id: int) -> str:
        """
        get most recent finetuned model from database or empty string
        character_id: current character
        :return: path to gguf
        """
        path = ""

        sql = "select * from finetuned_models where character_id = ? order by id desc"
        res = self.cur.execute(sql, (character_id,)).fetchall()
        for row in res:
            path = row["path"]
            if os.path.exists(path):
                return path
        return path

    def create_finetuned_model(self, character_id: int, qlora: bool = False) -> None:
        """
        Load the hf model and create a gguf model. Apply qlora if necassary.
        :param character_id: character id
        :param qlora: should qlora based on chat history be created?
        """
        if qlora:
            raise Exception()

        sql = "select name from characters where id = ?"
        res = self.cur.execute(sql, (character_id,)).fetchall()
        if len(res) > 0:
            charname = res[0]["name"]
        else:
            raise CharacterDoesntExistsException()

        path = self.gs.config["hf_model_path"]
        out_type = self.gs.config["gguf_quantization"]
        ctx = self.gs.config["context_size"]
        date = time.strftime("%Y%m%d_%H%M%S")

        outfile = f"finetune_{charname}_{date}.gguf"
        outfile_path = os.path.join(self.gs.config["model_path"], outfile)

        argv = [path, "--outtype", out_type, "--outfile", outfile_path, "--ctx", f"{ctx}"]

        convert.main(args_in=argv)

        if os.path.exists(outfile_path):
            sql = "insert into finetuned_models (character_id, path, time) values (?, ?, ?)"
            ts = datetime.now(timezone.utc)
            self.cur.execute(sql, (character_id, outfile_path, ts))
            self.con.commit()
        else:
            raise Exception("Unable to create file!")





