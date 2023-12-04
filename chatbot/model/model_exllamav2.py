import json, uuid, os, gc
import torch
from pynvml import *

from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Cache_8bit,
    ExLlamaV2Tokenizer,
)

from exllamav2.generator import(
    ExLlamaV2BaseGenerator,
    ExLlamaV2Sampler
)

from chatbot.global_state import GlobalState
from chatbot.model.model_base import ModelBase

class ModelExllamaV2(ModelBase):
    def __init__(self):
        super().__init__()

        self.gs = GlobalState()
        self.generator = None

        self.model_path = self.gs.config["exllamav2_model_path"]
        self.init_model()

    def init_model(self):
        model = {}
        model["model_directory"] = self.model_path
        model["seq_len"] = self.gs.config["context_size"]
        model["rope_scale"] = 1
        model["rope_alpha"] = 1
        model["chunk_size"] = 2048

        self.model_dict = {}
        self.model_dict["cache_mode"] = "FP8"
        self.model_dict["gpu_split_auto"] = True
        self.model_dict["gpu_split"] = None

        self.config = ExLlamaV2Config()
        self.config.model_dir = model["model_directory"]
        self.config.prepare()

        self.config.max_seq_len = model["seq_len"]
        self.config.scale_pos_emb = model["rope_scale"]
        self.config.scale_alpha_value = model["rope_alpha"]
        self.config.max_input_len = model["chunk_size"]
        self.config.max_attn_size = model["chunk_size"] ** 2

        if self.model_dict["cache_mode"] == "FP8": self.cache_fp8 = True
        elif self.model_dict["cache_mode"] == "FP16": self.cache_fp8 = False
        else: raise ValueError("bad cache_mode: " + self.model_dict["cache_mode"])

        self.tokenizer = ExLlamaV2Tokenizer(self.config)

        # Load model
        self.model = ExLlamaV2(self.config)
        print("Loading model: " + self.config.model_dir)

        if self.model_dict["gpu_split_auto"]:
            auto_split = True
        elif self.model_dict["gpu_split"] is None or self.model_dict["gpu_split"].strip() == "":
            auto_split = False
            split = None
        else:
            auto_split = False
            split = [float(alloc) for alloc in self.model_dict["gpu_split"].split(",")]

        if self.cache_fp8:
            self.cache = ExLlamaV2Cache_8bit(self.model, lazy = auto_split)
        else:
            self.cache = ExLlamaV2Cache(self.model, lazy = auto_split)

        self.model.load_autosplit(self.cache)

        # Test VRAM allocation with a full-length forward pass
        input_ids = torch.zeros((1, self.config.max_input_len), dtype = torch.long)
        self.model.forward(input_ids, cache = self.cache, preprocess_only = True)

        # Create generator
        self.generator = ExLlamaV2BaseGenerator(self.model, self.cache, self.tokenizer)

    def unload_model(self):
        if self.model:
            self.model.unload()
        self.model = None
        self.config = None
        self.cache = None
        self.tokenizer = None

        gc.collect()
        torch.cuda.empty_cache()

    def get_response(self, prompt: str, max_token_length: int, stop_words: [str]) -> str:
        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = 0.85
        settings.top_k = 50
        settings.top_p = 0.95
        settings.token_repetition_penalty = 1.15
        settings.disallow_tokens(self.tokenizer, [self.tokenizer.eos_token_id])

        ids = self.tokenizer.encode(prompt)
        tokens_prompt = ids.shape[-1]

        output = self.generator.generate_simple(prompt, settings, max_token_length, token_healing=True)
        torch.cuda.synchronize()

        print(output)

        if "</s>" in output:
            output = output.split("</s>")[0]
        output = output.strip()

        return output
