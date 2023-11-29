import copy
import time
import gc
import torch

from llama_cpp import Llama, llama_free_model, LogitsProcessor, LogitsProcessorList

from chatbot.global_state import GlobalState
from chatbot.model.model_base import ModelBase

class ModelGguf(ModelBase):
    def __init__(self, model_path: str):
        super().__init__()

        self.gs = GlobalState()
        self.llm = None

        self.model_path = model_path
        self.init_model()

    def init_model(self):
        self.llm = Llama(model_path=self.model_path,
                    n_ctx=self.gs.config["context_size"],
                    n_gpu_layers=self.gs.config["gguf_gpu_layers"],
                    seed=int(time.time()))

    def unload_model(self):
        del self.llm
        self.llm = None
        gc.collect()
        torch.cuda.empty_cache()

    def get_response(self, prompt: str, max_token_length: int, stop_words: [str]) -> str:
        #self.unload_model()
        #self.init_model()

        args = self.gs.config["gguf_generation_parameters"]
        tmp = copy.copy(args)

        if "temperature" in tmp:
            tmp["temperature"] = tmp["temperature"] + self.gs.temperature_modifier
        if "top_p" in tmp:
            tmp["top_p"] = tmp["top_p"] + self.gs.top_p_modifier

        output = self.llm(prompt,
                          max_tokens=max_token_length,
                          stop=stop_words,
                          echo=False,
                          **tmp)
        full_out = output["choices"][0]["text"].replace("\\n", "\n")
        return full_out
