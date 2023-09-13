class ModelBase:
    def __init__(self):
        super().__init__()

    def init_model(self):
        raise NotImplementedError()

    def get_response(self, prompt: str, max_token_length: int, stop_words: [str]) -> str:
        raise NotImplementedError()
