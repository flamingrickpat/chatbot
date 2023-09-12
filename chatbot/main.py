import logging
import os
from typing import List
from pathlib import Path

from chatbot.config import get_config
from chatbot.config import validate_arguments
from chatbot.logger import setup_logging_default, setup_logging_config

from chatbot.telegram import run_telegram_bot

from chatbot.constants import *

from chatbot.message_manager import MessageManager
from chatbot.model_manager import ModelManager
from chatbot.global_state import GlobalState
from chatbot.summary import SummaryOpenai
from chatbot.chroma_manager import ChromaManager

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

    if config["summarizer"] == "openai":
        summarizer = SummaryOpenai()
        summarizer.init_summarizer()
        gs.summarizer = summarizer

    message_manager = MessageManager()
    gs.message_manager = message_manager

    model_manager = ModelManager()
    gs.model_manager = model_manager

    chroma = ChromaManager()
    gs.chroma_manager = chroma

    # Make new chroma db on startup
    gs.message_manager.generate_missing_chroma_entries()

    gs.telegram_chat_id = 0
    gs.telegram_message_id = 0

    setup_logging_config(config, "logs/chatbot.log")

    logging.getLogger("httpx").setLevel(logging.WARNING)

    logger.info("Starting Telegram bot...")
    gs.telegram_state = TELEGRAM_STATE_UNINITIALIZED
    run_telegram_bot()
