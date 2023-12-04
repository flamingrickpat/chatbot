import logging
from chatbot.logger import setup_logging_default, setup_logging_config

from chatbot.telegram import run_telegram_bot

from chatbot.constants import *

from chatbot.message_manager import MessageManager
from chatbot.model_manager import ModelManager
from chatbot.global_state import GlobalState
from chatbot.summary import SummaryOpenai, SummaryBart
from chatbot.chroma_manager import ChromaManager
from chatbot.emotion_manager import EmotionManger
from chatbot.db_manager import DbManager
from chatbot.summary_manager import SummaryManager
from chatbot.concept_manager import ConceptManager

logger = logging.getLogger('init_chatbot')

def init_chatbot():
    gs = GlobalState()

    dbm = DbManager()
    gs.db_manager = dbm

    sm = SummaryManager()
    gs.summary_manager = sm

    em = EmotionManger()
    gs.emotion_manager = em

    message_manager = MessageManager()
    gs.message_manager = message_manager

    model_manager = ModelManager()
    gs.model_manager = model_manager

    model_manager.get_message("tell me a story: ", [])
    exit(1)

    chroma = ChromaManager()
    gs.chroma_manager = chroma

    cm = ConceptManager()
    gs.concept_manager = cm

    gs.telegram_chat_id = 0
    gs.telegram_message_id = 0

    setup_logging_config(gs.config, "logs/chatbot.log")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logger.info("Starting Telegram bot...")
    gs.telegram_state = TELEGRAM_STATE_UNINITIALIZED

    if gs.config["autoselect_character"] != "":
        message_manager.select_character(gs.config["autoselect_character"])
        gs.telegram_state = TELEGRAM_STATE_CHAT

    # Make new chroma db on startup
    #message_manager.generate_missing_nsfw_ratio()
    #gs.message_manager.generate_missing_chroma_entries()
    #em.recalc_emotions()
    #sm.recalc_summaries()
    #chroma.calc_embeddings_messages(id=None)
    #chroma.calc_embeddings_summaries(id=None)
    #cm.calc_concepts_summaries(id=None)

    #run_telegram_bot()
    #prompt = gs.message_manager.get_prompt()
    #print(prompt)

    #if gs.config["interface_type"] == "telegram":
    #    run_telegram_bot()
