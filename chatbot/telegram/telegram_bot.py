import logging
from typing import Dict, Any
from telegram import ForceReply, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, ConversationHandler, CallbackContext, CallbackQueryHandler
from queue import Queue
from chatbot.config import configuration


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}!",
        reply_markup=ForceReply(selective=True),
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("Help!")

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo the user message."""
    user = update.effective_user
    text = update.message.text
    await update.message.reply_text(text)

async def InlineKeyboardHandler(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    await query.answer(f'')

def run_telegram_bot(config: Dict[str, Any]):
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(config["telegram"]["api_key"]).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    application.add_handler(CallbackQueryHandler(InlineKeyboardHandler))
    # application.add_handler(CommandHandler('request_button', menu))

    # on non command i.e message - echo the message on Telegram
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)