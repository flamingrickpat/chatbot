config_schema = \
    {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://flamingrickp.at/chatbot.schema.json",
        "title": "chatbot",
        "description": "chatbot configuration file",
        "type": "object",
        "required": ["user_name", "context_size", "telegram", "tokenizer_path", "model_path", "device", "summarizer",
                     "summarizer_message_count"],
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
            "model_path": {
                "type": "string",
                "description": "path to model. should be a folder with .bin files",
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
