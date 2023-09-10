import logging
import os
from typing import List
from pathlib import Path

from chatbot.config import get_config
from chatbot.config import validate_arguments
from chatbot.logger import setup_logging_default, setup_logging_config

logger = logging.getLogger('chatbot')

def main(args: List[str]) -> None:
    """
    Parse the arguments and config file.
    Setup the logger.
    :param args: parameters
    """
    setup_logging_default()

    args = validate_arguments(args)

    config_path = args.config
    config = get_config(config_path)

    setup_logging_config(config, "logs/chatbot.log")

    logger.info("Test")