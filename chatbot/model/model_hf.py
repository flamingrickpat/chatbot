import random
import time
import gc

import torch
from peft import PeftModel
import torch
import torch.nn as nn
import transformers
from peft import (
    LoraConfig,
    PeftConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteriaList
)

from accelerate import infer_auto_device_map, init_empty_weights
from accelerate import load_checkpoint_and_dispatch
from datasets import load_dataset

from chatbot.global_state import GlobalState
from chatbot.model_utils import StoppingCriteriaSub
from chatbot.model.model_base import ModelBase


class Callbacks(transformers.TrainerCallback):
    def on_step_begin(self, args: transformers.TrainingArguments, state: transformers.TrainerState,
                      control: transformers.TrainerControl, **kwargs):
        pass

    def on_substep_end(self, args: transformers.TrainingArguments, state: transformers.TrainerState,
                       control: transformers.TrainerControl, **kwargs):
        pass

    def on_log(self, args: transformers.TrainingArguments, state: transformers.TrainerState,
               control: transformers.TrainerControl, logs, **kwargs):
        if 'loss' in logs:
            loss = float(logs['loss'])
            stop_at_loss = GlobalState().config["hf_lora_stop_at_loss"]
            if loss <= stop_at_loss:
                control.should_epoch_stop = True
                control.should_training_stop = True
                print(f"\033[1;31;1mStop Loss {stop_at_loss} reached.\033[0;37;0m")


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
        model_path = self.gs.config["hf_model_path"]
        quant = self.gs.config["hf_quant"]

        params = {}
        gpu_mem = self.gs.config["hf_max_gpu_memory"]
        cpu_mem = self.gs.config["hf_max_cpu_memory"]
        device_map = self.device

        if gpu_mem > 0 or cpu_mem > 0:
            if cpu_mem == 0:
                cpu_mem = 128
            max_memory = {0: f'{gpu_mem}GiB', 'cpu': f'{cpu_mem}GiB'}

            model = None
            config = AutoConfig.from_pretrained(model_path)
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(config)
            model.tie_weights()

            device_map = infer_auto_device_map(
                model,
                dtype=torch.int8,
                max_memory=max_memory,
                no_split_module_classes=model._no_split_modules
            )

            params["max_memory"] = {0: f'{gpu_mem}GiB', 'cpu': f'{cpu_mem}GiB'}
            device_map = "auto"


        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
            device_map=device_map
        )

        if quant == "":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
                device_map=device_map,
                early_stopping=True,
                low_cpu_mem_usage=True,
                **params
            )

            lora_path = self.gs.config["hf_lora_path"]
            if lora_path != "":
                self.model = PeftModel.from_pretrained(self.model,
                                                       self.gs.config["hf_lora_path"],
                                                       device_map=self.device)
                self.model = self.model.merge_and_unload()
        elif quant == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
                device_map=device_map,
                early_stopping=True,
                low_cpu_mem_usage=True,
                load_in_4bit=True,
                quantization_config=bnb_config,
                **params
            )
        elif quant == "8bit":
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_quant_type="nf8",
                bnb_8bit_compute_dtype=torch.float16,
                bnb_8bit_use_double_quant=True,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
                device_map=device_map,
                early_stopping=True,
                low_cpu_mem_usage=True,
                load_in_8bit=True,
                quantization_config=bnb_config,
                **params
            )

    def unload_model(self):
        del self.model
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()

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

        return output

    def lora(self, dataset_path: str, out_path: str) -> None:
        def generate_prompt(data_point):
            if self.gs.config["add_instructions"]:
                return f"""
{data_point["Context"]}

{data_point["Response"]}
""".replace("\\n", "\n").strip()
            else:
                return f"""
{data_point["Context"].replace("### Input:", "").strip()}

{data_point["Response"].replace("### Response:", "").strip()}
""".replace("\\n", "\n").strip()

        def generate_and_tokenize_prompt(data_point):
            full_prompt = generate_prompt(data_point)
            tokenized_full_prompt = self.tokenizer(full_prompt, padding=True, truncation=True)
            return tokenized_full_prompt

        dataset = load_dataset("json", data_files=dataset_path)
        dataset = dataset.shuffle().map(generate_and_tokenize_prompt)
        dataset = dataset["train"]

        self.model = prepare_model_for_kbit_training(self.model)

        config = LoraConfig(
            r=256,
            lora_alpha=512,
            target_modules=[
                "q_proj",
                "v_proj"
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.model = get_peft_model(self.model, config)

        batch_size = 128
        micro_batch_size = 4
        gradient_accumulation_steps = batch_size // micro_batch_size
        epochs = 10

        training_args = transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=epochs,
            learning_rate=3e-5,
            bf16=True,
            save_total_limit=1,
            logging_steps=2,
            save_strategy='no',
            output_dir=out_path
        )

        trainer = transformers.Trainer(
            model=self.model,
            train_dataset=dataset,
            args=training_args,
            data_collator=transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            callbacks=list([Callbacks()])
        )

        self.model.config.use_cache = False
        trainer.train()

        self.model.save_pretrained(out_path)
        time.sleep(10)

