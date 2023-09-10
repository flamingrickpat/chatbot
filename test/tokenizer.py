import time
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

timeStart = time.time()

model_name = "X:/AI/oobabooga2/oobabooga_windows/text-generation-webui/models/airoboros-l2-7b-2.1"
adapters_name = "X:/AI/oobabooga2/oobabooga_windows/text-generation-webui/loras/kara"
device = "cpu"

t = time.time()
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    local_files_only=True,
    device_map=device
)
t1 = time.time()
inputs = tokenizer.encode(
    "ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.",
    return_tensors="pt"
).to(device)

print(inputs.shape[1])

t2 = time.time()

print(t1 - t)
print(t2 - t1)