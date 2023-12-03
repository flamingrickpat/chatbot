import logging
from typing import List
from chatbot.config import get_config
from chatbot.config import validate_arguments
from chatbot.logger import setup_logging_default, setup_logging_config
from chatbot.global_state import GlobalState
from chatbot.webui import start_webui
from chatbot.init_chatbot import init_chatbot
from chatbot.telegram.telegram_bot import run_telegram_bot

logger = logging.getLogger('chatbot')

def main(args: List[str]) -> None:
    """
    Parse the arguments and config file.
    Setup the logger.
    Start message manager and telegram bot.
    :param args: parameters
    """
    gs = GlobalState()

    setup_logging_default()

    args = validate_arguments(args)

    config_path = args.config
    config = get_config(config_path)
    gs.config = config
    gs.temperature_modifier = 0
    gs.top_p_modifier = 0
    gs.regenerate_counter = 0

    init_chatbot()

    if config["interface_type"] == "telegram":
        run_telegram_bot()
    elif config["interface_type"] == "webui":
        start_webui()

