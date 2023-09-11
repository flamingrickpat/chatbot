import time
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

timeStart = time.time()

model_name = "X:/AI/oobabooga2/oobabooga_windows/text-generation-webui/models/airoboros-l2-7b-2.1"
adapters_name = "X:/AI/oobabooga2/oobabooga_windows/text-generation-webui/loras/kara"
device = "cuda:0"


tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    local_files_only=True,
    device_map=device
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    device_map=device
)


#model = PeftModel.from_pretrained(model, adapters_name)
#model = model.merge_and_unload()

print(f"Successfully loaded the model {model_name} into memory")


print("Load model time: ", -timeStart + time.time())

while(True):
    input_str = input('Enter: ')
    input_token_length = input('Enter length: ')

    if(input_str == 'exit'):
        break

    timeStart = time.time()

    inputs = tokenizer.encode(
        input_str,
        return_tensors="pt"
    ).to(device)

    outputs = model.generate(
        inputs,
        max_new_tokens=int(input_token_length),
    )

    output_str = tokenizer.decode(outputs[0])

    print(output_str)

    print("Time taken: ", -timeStart + time.time())


del model
torch.cuda.empty_cache()

while True:
    input_str = input('Enter: ')
    if input_str == "exit":
        quit()