import logging
import time
from typing import Dict, Any
from telegram import ForceReply, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram._utils.defaultvalue import DEFAULT_NONE
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, ConversationHandler, CallbackContext, CallbackQueryHandler
from queue import Queue
from chatbot.config import configuration
from chatbot.global_state import GlobalState
from chatbot.exceptions import *

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}!\nType /help to see available commands.",
        reply_markup=ForceReply(selective=True),
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("""
/help - Show available commands
/list - Show all characters.
/add {character_name} - Create a new character
/delete {character_name} - Delete character
/select {character_name} - Select current character
/card - Change character card
/regenerate - Regenerate last reply
""")


async def list_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List available characters"""
    gs = GlobalState()
    chars = gs.message_manager.list_available_characters()
    res = ""
    for char in chars:
        res = res + char + "\n"
    await update.message.reply_text(res.strip())

async def add_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Create a new character."""
    gs = GlobalState()
    try:
        name = context.args[0]
        gs.message_manager.add_character(name)
        await update.message.reply_text(f"Successfully created character {name}!")
    except CharacterAlreadyExistsException as e:
        await update.message.reply_text("Character already exists!")
    except IndexError as e:
        await update.message.reply_text("You need to submit the name as parameter!")
    except Exception as e:
        await update.message.reply_text("Error: " + str(e))


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo the user message."""
    user = update.effective_user
    text = update.message.text
    tmp = await update.message.reply_text(text)

    chat_id = tmp.chat_id
    message_id = tmp.message_id

    time.sleep(5)

    await context.bot.deleteMessage(
        chat_id=chat_id,
        message_id=message_id,
        read_timeout=DEFAULT_NONE,
        write_timeout=DEFAULT_NONE,
        connect_timeout=DEFAULT_NONE,
        pool_timeout=DEFAULT_NONE,
        api_kwargs=None
    )


async def InlineKeyboardHandler(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    await query.answer(f'')

def run_telegram_bot() -> None:
    """Start the bot."""

    gs = GlobalState()

    # Create the Application and pass it your bot's token.
    application = Application.builder().token(gs.config["telegram"]["api_key"]).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("list", list_command))
    application.add_handler(CommandHandler("add", add_command))

    application.add_handler(CallbackQueryHandler(InlineKeyboardHandler))
    # application.add_handler(CommandHandler('request_button', menu))

    # on non command i.e message - echo the message on Telegram
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)