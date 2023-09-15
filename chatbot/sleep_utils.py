import logging
import copy
import gc
import time
import os
import sqlite3
from datetime import datetime, timezone

import jsonlines
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from chatbot.global_state import GlobalState
from chatbot.model import convert


logger = logging.getLogger('sleep_utils')

def format_full_history(filename: str, out_path: str) -> None:
    ds = []
    cur_object = {}

    mode = 0
    old_mode = 0

    cur_input = ""

    with open(filename, encoding="utf8") as file:

        cur_dict = None
        while line := file.readline():
            l = line.rstrip()

            if "### Input:" in l:
                mode = 0

            if "### Response:" in l:
                mode = 1

            if mode != old_mode:
                if mode == 1:
                    cur_object["Context"] = cur_input.strip()
                    cur_input = ""
                elif mode == 0:
                    cur_object["Response"] = cur_input.strip()
                    ds.append(copy.copy(cur_object))
                    cur_input = ""
                    cur_object = {}

            cur_input = cur_input + l + "\n"
            old_mode = mode

    with jsonlines.open(out_path, mode='w') as writer:
        writer.write_all(ds)


def load_merge_save(model_path: str, lora_path: str, out_path: str) -> None:
    gs = GlobalState()
    device = gs.config["hf_device"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        device_map=device
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        device_map=device,
    )

    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()

    tokenizer.save_pretrained(out_path)
    model.save_pretrained(out_path)

    time.sleep(10)

    del model
    model = None
    gc.collect()
    torch.cuda.empty_cache()


def convert_to_gguf(input_path: str, output_path: str, character_id: int):
    gs = GlobalState()

    out_type = gs.config["gguf_quantization"]
    ctx = gs.config["context_size"]
    date = time.strftime("%Y%m%d_%H%M%S")

    argv = [input_path, "--outtype", out_type, "--outfile", output_path, "--ctx", f"{ctx}"]

    convert.main(args_in=argv)

    if os.path.exists(output_path):
        con = sqlite3.connect(gs.config["database_path"])
        con.row_factory = sqlite3.Row
        cur = con.cursor()

        sql = "insert into finetuned_models (character_id, base_model, path, time) values (?, ?, ?, ?)"
        ts = datetime.now(timezone.utc)
        cur.execute(sql, (character_id, gs.config["hf_model_path"], output_path, ts))
        con.commit()
    else:
        raise Exception("Unable to create file!")
