import transformers
import torch
import asyncio
import telegram
import threading

from chatbot.global_state import GlobalState


async def something_async(output, telegram_chat_id, telegram_message_id):
    gs = GlobalState()
    bot = telegram.Bot(token=gs.config["telegram"]["api_key"])
    for user_id in gs.config["telegram"]["user_whitelist"]:
        try:
            await bot.editMessageText(
                chat_id=gs.telegram_chat_id,
                message_id=gs.telegram_message_id,
                text=output
            )
        except Exception as e:
            pass


class StoppingCriteriaSub(transformers.StoppingCriteria):
    def __init__(self, stop_strings=None, prompt_length=0, tokenizer=None, telegram_chat_id=0, telegram_message_id=0,
                 telegram_context=None):
        super().__init__()
        if stop_strings is None:
            stop_strings = []
        self.stop_strings = stop_strings
        self.prompt_length = prompt_length
        self.tokenizer = tokenizer
        self.telegram_chat_id = telegram_chat_id
        self.telegram_message_id = telegram_message_id
        self.telegram_context = telegram_context

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        output = self.tokenizer.decode(input_ids[0][self.prompt_length:])

        if False:
            thread = threading.Thread(target=asyncio.run, args=(something_async(output, self.telegram_chat_id, self.telegram_message_id),))
            thread.start()

        for stop in self.stop_strings:
            if stop in output:
                return True

        return False
