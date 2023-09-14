import time

from llama_cpp import Llama

from chatbot.global_state import GlobalState
from chatbot.model.model_base import ModelBase


class ModelGguf(ModelBase):
    def __init__(self, model_path: str):
        super().__init__()

        self.gs = GlobalState()

        self.model_path = model_path
        self.init_model()

    def init_model(self):
        self.llm = Llama(model_path=self.model_path,
                    n_ctx=self.gs.config["context_size"],
                    n_gpu_layers=self.gs.config["gguf_gpu_layers"])

    def get_response(self, prompt: str, max_token_length: int, stop_words: [str]) -> str:
        output = self.llm(prompt, max_tokens=max_token_length, stop=stop_words, echo=False)
        full_out = output["choices"][0]["text"].replace("\\n", "\n")
        return full_out
