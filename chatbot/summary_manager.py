from transformers import pipeline
from datetime import datetime, timezone

from chatbot.summary import SummaryOpenai, SummaryBart
from chatbot.global_state import GlobalState

class SummaryManager:
    def __init__(self):
        self.emotion_classifier = None
        self.nsfw_classifier = None
        self.gs = GlobalState()
        self.con = self.gs.db_manager.con
        self.cur = self.gs.db_manager.cur

        if self.gs.config["summarizer"] == "openai":
            summarizer = SummaryOpenai()
            summarizer.init_summarizer()
            self.summarizer = summarizer
        elif self.gs.config["summarizer"] == "bart":
            summarizer = SummaryBart()
            summarizer.init_summarizer()
            self.summarizer = summarizer

    def summarize_last_messages(self, current_character_id: int) -> None:
        """
        Call summarizer, summarize last x messages and write it to summaries table.
        """

        if self.gs.config["summarizer_omit_nsfw"]:
            res = self.cur.execute("SELECT * FROM messages where character_id = ? and nsfw_ratio < ?",
                                   (current_character_id, self.gs.config["summarizer_omit_nsfw_cutoff"]))
        else:
            res = self.cur.execute("SELECT * FROM messages where character_id = ?", (current_character_id,))
        res = res.fetchall()

        text = ""

        in_window = res[max(0, len(res) - self.gs.config["summarizer_message_count"]):]
        ids = []
        for msg in in_window:
            ids.append(msg["id"])
            text = text + msg["character"] + ": " + msg["message"] + "\n"

        summary = self.summarizer.summarize_text(text)
        token_count = self.gs.model_manager.get_token_count(summary)

        send_date = datetime.now(timezone.utc)
        self.cur.execute("INSERT INTO summaries (character_id, summary, time, token_count) VALUES(?, ?, ?, ?)",
                         (current_character_id, summary, send_date, token_count))
        self.con.commit()
        inserted_id = self.cur.lastrowid

        # insert into chroma
        self.gs.chroma_manager.insert(is_message=False, id=inserted_id, character_id=current_character_id,
                                      is_user=False, text=summary, token_count=token_count)

        # Insert relation to messages
        for msg_id in ids:
            self.cur.execute("INSERT INTO summaries_messages (message_id, summary_id) VALUES(?, ?)",
                             (msg_id, inserted_id))
            self.con.commit()

    def recalc_summaries(self):
        self.cur.execute("delete from summaries")
        self.cur.execute("delete from summaries_messages")
        self.con.commit()

