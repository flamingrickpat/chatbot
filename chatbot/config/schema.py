config_schema = \
    {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://flamingrickp.at/chatbot.schema.json",
        "title": "kektrade",
        "description": "kektrade metastrategy configuration file",
        "type": "object",
        "required": ["telegram"],
        "properties": {
            "telegram": {
                "$ref": "#/$defs/telegram",
                "description": "telegram settings"
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
