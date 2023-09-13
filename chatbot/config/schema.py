config_schema = \
    {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://flamingrickp.at/chatbot.schema.json",
        "title": "chatbot",
        "description": "chatbot configuration file",
        "type": "object",
        "required": ["user_name", "context_size", "telegram", "tokenizer_path", "summarizer",
                     "summarizer_message_count", "model"],
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
                "enum": ["hf", "api"]
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
