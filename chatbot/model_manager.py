import os
from typing import Dict, List, Any, Tuple
import time
import logging
import sqlite3
from datetime import datetime, timezone
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizerFast, StoppingCriteriaList

from chatbot.global_state import GlobalState
from chatbot.exceptions import *

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

        self.con = self.gs.db_manager.con
        self.cur = self.gs.db_manager.cur

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
            from chatbot.model.model_hf import ModelHf
            self.model = ModelHf()
        elif model_type == "api":
            from chatbot.model.model_api import ModelApi
            self.model = ModelApi()
        elif model_type == "gguf":
            from chatbot.model.model_gguf import ModelGguf
            self.model = ModelGguf(model_type)
        elif model_type == "exllamav2":
            from chatbot.model.model_exllamav2 import ModelExllamaV2
            self.model = ModelExllamaV2()

    def unload_model(self) -> None:
        """Unload model and free VRAM."""
        if self.model is not None:
            self.model.unload_model()

    def reload_model(self, character_id: int) -> None:
        """
        Reload model when switching character.
        :param character_id:
        :return:
        """

        logger.info(f"Reloading model for {character_id}...")

        model_type = self.gs.config["model"]
        if model_type == "gguf":
            from chatbot.model import ModelGguf

            path = self.get_finetuned_model_path(character_id)
            if path == "":
                logger.info(f"Creating first model")
                self.create_first_gguf_model(character_id)
                path = self.get_finetuned_model_path(character_id)

            logger.info(f"Using model: {path}")
            self.model = ModelGguf(path)


    def get_message(self, prompt: str, stop_words: [str]) -> str:
        result = self.model.get_response(prompt, 256, stop_words)
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

        sql = "select * from finetuned_models where character_id = ? and base_model = ? order by id desc"
        res = self.cur.execute(sql, (character_id, self.gs.config["hf_model_path"])).fetchall()
        for row in res:
            path = row["path"]
            if os.path.exists(path):
                return path
        return path

    def create_first_gguf_model(self, character_id: int) -> None:
        """
        Load the hf model and create a gguf model. Apply qlora if necassary.
        :param character_id: character id
        :param lora: should lora based on chat history be created?
        """
        sql = "select name from characters where id = ?"
        res = self.cur.execute(sql, (character_id,)).fetchall()
        if len(res) > 0:
            charname = res[0]["name"]
        else:
            raise CharacterDoesntExistsException()

        path = self.gs.config["hf_model_path"]
        date = time.strftime("%Y%m%d_%H%M%S")

        outfile = f"finetune_{charname}_{date}.gguf"
        outfile_path = os.path.join(self.gs.config["model_path"], outfile)

        from chatbot import sleep_utils
        sleep_utils.convert_to_gguf(input_path=path, output_path=outfile_path, character_id=character_id)

    def sleep(self, character_id: int):
        gs = GlobalState()
        model_manager = gs.model_manager
        message_manager = gs.message_manager

        sql = "select name from characters where id = ?"
        res = self.cur.execute(sql, (character_id,)).fetchall()
        if len(res) > 0:
            charname = res[0]["name"]
        else:
            raise CharacterDoesntExistsException()

        model_path = gs.config["hf_model_path"]
        temp_finetune_path = gs.config["temp_finetune_path"]
        temp_full_history_path = os.path.join(temp_finetune_path, "full_history.txt")
        temp_full_history_format_path = os.path.join(temp_finetune_path, "full_history_format.jsonl")
        temp_lora_path = os.path.join(temp_finetune_path, "temp_lora")
        tmp_merge_path = os.path.join(temp_finetune_path, "temp_merged")

        logger.info("Cleaning temp folder...")
        if os.path.exists(temp_finetune_path):
            shutil.rmtree(temp_finetune_path)
        os.mkdir(temp_finetune_path)

        logger.info("Unloading gguf model...")
        model_manager.unload_model()

        time.sleep(10)

        quant = gs.config["hf_quant"]
        logger.info(f"Loading hf model with quant level: {quant}...")
        model = ModelHf()

        logger.info("Making dataset...")
        messages = message_manager.get_messages_for_lora()
        with open(temp_full_history_path, "w", encoding="utf-8") as f:
            f.write(messages)
        sleep_utils.format_full_history(temp_full_history_path, temp_full_history_format_path)

        logger.info("Training QLoRA...")
        model.lora(dataset_path=temp_full_history_format_path, out_path=temp_lora_path)

        logger.info("Reload hf model without quant")
        model.unload_model()

        logger.info("Merge and save to temp folder...")
        sleep_utils.load_merge_save(model_path=model_path, lora_path=temp_lora_path, out_path=tmp_merge_path)

        logger.info("Convert hf model to gguf and save to models path...")
        date = time.strftime("%Y%m%d_%H%M%S")
        outfile = f"finetune_{charname}_{date}.gguf"
        outfile_path = os.path.join(self.gs.config["model_path"], outfile)
        sleep_utils.convert_to_gguf(input_path=tmp_merge_path, output_path=outfile_path, character_id=character_id)

        logger.info("Load new gguf model...")
        self.reload_model(character_id=character_id)

        logger.info("Sleep successful!")




