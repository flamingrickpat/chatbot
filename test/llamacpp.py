from llama_cpp import Llama

prompt = ""
with open("X:\\Workspace\\Repositories\\chatbot_new\\logs\\latest_prompt.txt", 'r', encoding="utf-8") as file:
    prompt = file.read()

llm = Llama(model_path="X:\\AI\\athena_merge_test.gguf", n_ctx=4096, n_gpu_layers=100)

output = llm(prompt, max_tokens=250, stop=[], echo=False)
print(output["choices"][0]["text"].replace("\\n", "\n"))



