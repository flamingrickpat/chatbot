import transformers
import torch

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

        if self.telegram_context is not None:
            pass

        for stop in self.stop_strings:
            if stop in output:
                return True

        return False