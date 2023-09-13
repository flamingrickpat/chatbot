import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizerFast, \
    StoppingCriteriaList, BitsAndBytesConfig
import random
import time

from chatbot.global_state import GlobalState
from chatbot.model_utils import StoppingCriteriaSub
from chatbot.model.model_base import ModelBase




class ModelHf(ModelBase):
    def __init__(self):
        super().__init__()

        self.gs = GlobalState()
        self.device = self.gs.config["hf_device"]

        self.model = None
        self.tokenizer = None

        self.init_model()

    def init_model(self):
        tokenizer_path = self.gs.config["hf_model_path"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
            device_map=self.device
        )

        model_path = self.gs.config["hf_model_path"]
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            device_map=self.device,
            early_stopping=True,
        )

        lora_path = self.gs.config["hf_lora_path"]
        if lora_path != "":
            self.model = PeftModel.from_pretrained(self.model,
                                                   self.gs.config["hf_lora_path"],
                                                   device_map=self.device)
            self.model = self.model.merge_and_unload()

    def get_response(self, prompt: str, max_token_length: int, stop_words: [str]) -> str:
        seeds = int(time.time() * 1000)
        random.seed(seeds)
        torch.manual_seed(seeds)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seeds)

        tokenized = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        stopping_criteria_list = StoppingCriteriaList([StoppingCriteriaSub(stop_strings=stop_words,
                                                                           prompt_length=tokenized.input_ids.shape[1],
                                                                           tokenizer=self.tokenizer)])

        token = self.model.generate(**tokenized,
                                    max_new_tokens=max_token_length,
                                    do_sample=True,
                                    temperature=0.9,
                                    repetition_penalty=1.1,
                                    # eos_token_id=[],
                                    stopping_criteria=stopping_criteria_list,
                                    early_stopping=True)

        output = self.tokenizer.decode(token[0][tokenized.input_ids.shape[1]:])
        output = output.strip()

        # Remove emojis
        output = output.encode('ascii', 'ignore').decode('ascii').strip()

        return output

