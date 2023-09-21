config_schema = \
    {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://flamingrickp.at/chatbot.schema.json",
        "title": "chatbot",
        "description": "chatbot configuration file",
        "type": "object",
        "required": ["user_name", "context_size", "telegram", "tokenizer_path", "summarizer",
                     "summarizer_message_count", "model", "hf_quant"],
        "properties": {
            "telegram": {
                "$ref": "#/$defs/telegram",
                "description": "telegram settings"
            },
            "user_name": {
                "type": "string",
                "description": "name of user",
            },
            "context_size": {
                "type": "integer",
                "description": "how long prompts should be",
            },
            "max_jaro_distance": {
                "type": "number",
                "description": "max jaro distance to prevent model from generating too similar responses",
                "default": 0.8
            },
            "device": {
                "type": "string",
                "description": "what device to run model on",
                "enum": ["cpu", "cuda:0"]
            },
            "database_path": {
                "type": "string",
                "description": "path to database",
                "default": "./database/database.db"
            },
            "prompt_path": {
                "type": "string",
                "description": "path to latest promps for debugging",
                "default": "./logs/latest_prompt.txt"
            },
            "log_level": {
                "type": "string",
                "description": "minimum importance to show log line",
                "default": "info",
                "enum": ["debug", "info", "warning", "error", "critical"]
            },
            "log_console": {
                "type": "boolean",
                "description": "write log to stdout",
                "default": True
            },
            "tokenizer_path": {
                "type": "string",
                "description": "path to tokenizer. should be a folder with tokenizer.json",
            },
            "summarizer": {
                "type": "string",
                "description": "what summarizer should be used?",
                "enum": ["openai", "bart", "api_endpoint"]
            },
            "openai_api_key": {
                "type": "string",
                "description": "api key for openai, to use gpt3.5 as summarizer",
            },
            "bart_model": {
                "type": "string",
                "description": "path to bart model as summarizer",
            },
            "api_endpoint": {
                "type": "string",
                "description": "ip and port of koboldcpp or similar to use as summarizer",
            },
            "summarizer_message_count": {
                "type": "integer",
                "description": "how many of the previous messages should be summarized",
            },
            "chromadb_path": {
                "type": "string",
                "description": "path to chroma db embeddings",
                "default": "./database/chroma/"
            },
            "chromadb_embedder": {
                "type": "string",
                "description": "custom embedder",
                "default": ""
            },
            "memory_summary_length": {
                "type": "integer",
                "description": "max token allowance for memories in summary form",
                "default": 500
            },
            "memory_message_length": {
                "type": "integer",
                "description": "max token allowance for memories in message form",
                "default": 500
            },
            "model": {
                "type": "string",
                "description": "type of model to use in background",
                "enum": ["gguf", "hf", "api"]
            },
            "hf_model_path": {
                "type": "string",
                "description": "path to hf model"
            },
            "hf_lora_path": {
                "type": "string",
                "description": "path to hf lora if it exists, otherwise blank"
            },
            "hf_device": {
                "type": "string",
                "description": "what device to use for hf model. can be cpu or cuda:0"
            },
            "hf_quant": {
                "type": "string",
                "description": "quantization of hf model. can be left empty",
                "enum": ["", "4bit", "8bit"]
            },
            "api_url": {
                "type": "string",
                "description": "url for api you're using, can be oobabooga or koboldcpp"
            },
            "api_exe": {
                "type": "string",
                "description": "path to exe for starting up api"
            },
            "autoselect_character": {
                "type": "string",
                "description": "the character to select on startup. can be left empty",
                "default": ""
            },
            "greeting_message": {
                "type": "boolean",
                "description": "should bot notify in telegram that it is running?",
                "default": True
            },
            "hf_max_gpu_memory": {
                "type": "integer",
                "description": "how much gpu memory can me used. for accelerate. 0 for unlimited",
                "default": 0
            },
            "hf_max_cpu_memory": {
                "type": "integer",
                "description": "how much cpu memory can me used. for accelerate. 0 for unlimited",
                "default": 0
            },
            "message_streaming": {
                "type": "boolean",
                "description": "stream parts of messages to telegram",
                "default": False
            },
            "gguf_quantization": {
                "type": "string",
                "description": "quantization level for gguf creation",
                "default": "q8_0",
                "enum": ["f32", "f16", "q8_0"]
            },
            "model_path": {
                "type": "string",
                "description": "folder to store finetuned models",
                "default": "./models/"
            },
            "temp_finetune_path": {
                "type": "string",
                "description": "folder to store unquantized finetuned models and temporary files in",
                "default": "./models/temp"
            },
            "gguf_gpu_layers": {
                "type": "integer",
                "description": "how many gpu layers should be loaded by llama.cpp",
                "default": 128
            },
            "gguf_generation_parameters": {
                "type": "object",
                "description": "",
                "default": {}
            },
            "ascii_only": {
                "type": "boolean",
                "description": "remove non ascii characters from output",
                "default": False
            },
            "auto_raise_temperature": {
                "type": "number",
                "description": "if messages are too similar to other messages, raise the temperature by this much",
                "default": 0.02
            },
            "auto_raise_top_p": {
                "type": "number",
                "description": "if messages are too similar to other messages, raise the top_p by this much",
                "default": 0.02
            },
            "add_instructions": {
                "type": "boolean",
                "description": "add ###Input and ###Response to prompt",
                "default": True
            },
            "nsfw_classifier": {
                "type": "string",
                "description": "hf name or path to nsfw classifier",
                "default": "michellejieli/NSFW_text_classifier"
            },
            "emotion_classifier": {
                "type": "string",
                "description": "hf name or path to emotion classifier",
                "default": "nateraw/bert-base-uncased-emotion"
            },
            "banned_phrases": {
                "type": "array",
                "description": "phrases that must not be generated",
                "items": {"type": "string"},
                "default": []
            },
            "summarizer_omit_nsfw": {
                "type": "boolean",
                "description": "don't summarize messages that are rated nsfw",
                "default": True
            },
            "summarizer_omit_cutoff": {
                "type": "number",
                "description": "max nsfw ratio to ban from summarizer",
                "default": 0.9
            }
        },

        "$defs": {
            "telegram": {
                "type": "object",
                "required": ["api_key", "user_whitelist"],
                "properties": {
                    "api_key": {
                        "type": "string",
                        "description": "your telegram bot api key"
                    },
                    "user_whitelist": {
                        "type": "array",
                        "description": "whitelisted user ids",
                        "items": {"type": "integer"}
                    }
                }
            }
        }
    }
