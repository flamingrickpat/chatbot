config_schema = \
    {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://flamingrickp.at/chatbot.schema.json",
        "title": "chatbot",
        "description": "chatbot configuration file",
        "type": "object",
        "required": ["user_name", "context_size", "telegram", "tokenizer_path", "model_path", "device"],
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
