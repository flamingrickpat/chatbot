from transformers import LlamaTokenizer, LlamaForCausalLM
import transformers
import torch
import random


preprompt = '''Utachi: hello
pooh: i am hanging out on discord, how are you?
Utachi: My name is utachi and I'm so excited to meet you all! I love exploring new things and trying out new hobbies. Do you have any recommendations for what I should try next?
pooh: Hi nice to meet you! i am pooh nice to meet you. Are you interesting in watching anime? i am watching this new show call Bochi the rock it basiclly k-on but for people with social anxiety.
Utachi: '''

model = 'Neko-Institute-of-Science/pygmalion-7b'
model = "X:/AI/oobabooga2/oobabooga_windows/text-generation-webui/models/airoboros-l2-7b-2.1"

tokenizer = LlamaTokenizer.from_pretrained(model, local_files_only=True, device_map='cuda:0')
model = LlamaForCausalLM.from_pretrained(model, local_files_only=True, low_cpu_mem_usage=True, device_map='cuda:0',
                                         early_stopping=True)

tokenized = tokenizer(preprompt, return_tensors="pt").to('cuda:0')

seeds = 56416
random.seed(seeds)
torch.manual_seed(seeds)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seeds)

class StoppingCriteriaSub(transformers.StoppingCriteria):

    def __init__(self, stops=[], stop_strings=[], prompt_length=0, encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda:0") for stop in stops]
        self.stop_strings = stop_strings
        self.prompt_length = prompt_length

        print(self.stops)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        #print(input_ids)
        output = tokenizer.decode(input_ids[0][self.prompt_length:])
        print(output)
        for stop in self.stop_strings:
            if stop in output:
                return True

        #print(output)
        #for stop in self.stops:
        #    if torch.all((stop == input_ids[0][-len(stop):])).item():
        #        return True

        return False


stop_words = ["pooh:"]
stop_words_ids = [tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze()[1:] for stop_word in stop_words]
stopping_criteria_list = transformers.StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids, stop_strings=stop_words, prompt_length=tokenized.input_ids.shape[1])])

token = model.generate(**tokenized,
                       max_new_tokens=100,
                       do_sample=True,
                       eos_token_id=[],
                       stopping_criteria=stopping_criteria_list,
                       early_stopping=True)

output = tokenizer.batch_decode(token[:, tokenized.input_ids.shape[1]:])[0]
for word in stop_words:
    output = output.replace(word, "")
output = output.strip()

print(output)
