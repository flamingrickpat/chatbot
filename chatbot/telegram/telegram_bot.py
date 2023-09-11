import logging
import time
from typing import Dict, Any
import asyncio
from queue import Queue

import telegram
from telegram import ForceReply, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram._utils.defaultvalue import DEFAULT_NONE
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, ConversationHandler, CallbackContext, CallbackQueryHandler

from chatbot.config import configuration
from chatbot.global_state import GlobalState
from chatbot.exceptions import *
from chatbot.constants import *

def check_user(user_id: int) -> bool:
    gs = GlobalState()
    return user_id in gs.config["telegram"]["user_whitelist"]

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""

    if check_user(update.effective_user.id):
        user = update.effective_user
        await update.message.reply_html(
            rf"Hi {user.mention_html()}!\nType /help to see available commands.",
            reply_markup=ForceReply(selective=True),
        )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""

    if check_user(update.effective_user.id):
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

    if check_user(update.effective_user.id):
        gs = GlobalState()
        gs.telegram_state = TELEGRAM_STATE_UNINITIALIZED
        chars = gs.message_manager.list_available_characters()
        if len(chars) == 0:
            await update.message.reply_text("No characters!")
        else:
            res = ""
            for char in chars:
                res = res + char + "\n"
            await update.message.reply_text(res.strip())

async def add_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Create a new character."""

    if check_user(update.effective_user.id):
        gs = GlobalState()
        gs.telegram_state = TELEGRAM_STATE_UNINITIALIZED
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


async def delete_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Delete a character."""

    if check_user(update.effective_user.id):
        gs = GlobalState()
        gs.telegram_state = TELEGRAM_STATE_UNINITIALIZED
        try:
            name = context.args[0]
            gs.message_manager.delete_character(name)
            await update.message.reply_text(f"Successfully deleted character {name}!")
        except CharacterDoesntExistsException as e:
            await update.message.reply_text("Character doesn't exists!")
        except IndexError as e:
            await update.message.reply_text("You need to submit the name as parameter!")
        except Exception as e:
            await update.message.reply_text("Error: " + str(e))

async def select_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Delete a character."""

    if check_user(update.effective_user.id):
        gs = GlobalState()
        gs.telegram_state = TELEGRAM_STATE_UNINITIALIZED
        try:
            name = context.args[0]
            gs.message_manager.select_character(name)
            gs.telegram_state = TELEGRAM_STATE_CHAT
            await update.message.reply_text(f"Successfully selected character {name}!")
        except CharacterDoesntExistsException as e:
            await update.message.reply_text("Character doesn't exists!")
        except IndexError as e:
            await update.message.reply_text("You need to submit the name as parameter!")
        except Exception as e:
            await update.message.reply_text("Error: " + str(e))

async def card_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Change a character card."""

    if check_user(update.effective_user.id):
        gs = GlobalState()
        if gs.telegram_state == TELEGRAM_STATE_CHAT:
            gs.telegram_state = TELEGRAM_STATE_CARD
            await update.message.reply_text("Send the new character card with the next message!")
        else:
            await update.message.reply_text("Please select a character first!")

async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Chat with the bot.
    First, check if in the right state. Otherwise the user must select a character first.
    Then append user message to chat history in database and generate a prompt that is sent to the model.
    """

    if check_user(update.effective_user.id):
        gs = GlobalState()
        if gs.telegram_state == TELEGRAM_STATE_UNINITIALIZED:
            await update.message.reply_text("Please select a character first!")
        elif gs.telegram_state == TELEGRAM_STATE_CHAT:
            user = update.effective_user
            text = update.message.text

            gs.message_manager.insert_message(is_user=True, message=text)

            # Post message that is later edited
            msg = await update.message.reply_text("Thinking...")
            chat_id = msg.chat_id
            message_id = msg.message_id

            # Set parameters for live chat update
            gs.telegram_chat_id = chat_id
            gs.telegram_message_id = message_id

            # Get response
            db_id, response = gs.message_manager.get_response()
            gs.message_manager.set_telegram_info(db_id, chat_id, message_id)

            await asyncio.sleep(1)

            # Final edit
            await context.bot.editMessageText(
                chat_id=chat_id,
                message_id=message_id,
                text=response
            )

            return

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
        elif gs.telegram_state == TELEGRAM_STATE_CARD:
            text = update.message.text.strip()
            name, token_length = gs.message_manager.update_character_card(text)
            await update.message.reply_text(f"Character card for {name} was updated. Token length: {token_length}")
            gs.telegram_state = TELEGRAM_STATE_CHAT
    else:
        await update.message.reply_text("You are not white-listed and can't use this bot!")


async def InlineKeyboardHandler(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    await query.answer(f'')

def run_telegram_bot() -> None:
    """Start the bot."""

    gs = GlobalState()
    bot = telegram.Bot(token=gs.config["telegram"]["api_key"])
    for user_id in gs.config["telegram"]["user_whitelist"]:
        loop = asyncio.get_event_loop()
        loop.create_task(bot.send_message(chat_id=user_id, text='Chatbot has started up! '
                                                                'Please select a character to start chatting!'))

    # Create the Application and pass it your bot's token.
    application = Application.builder().token(gs.config["telegram"]["api_key"]).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("list", list_command))
    application.add_handler(CommandHandler("add", add_command))
    application.add_handler(CommandHandler("delete", delete_command))
    application.add_handler(CommandHandler("select", select_command))
    application.add_handler(CommandHandler("card", card_command))

    application.add_handler(CallbackQueryHandler(InlineKeyboardHandler))
    # application.add_handler(CommandHandler('request_button', menu))

    # on non command i.e message - echo the message on Telegram
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)