from transformers import pipeline
from datetime import datetime, timezone

from chatbot.summary import SummaryOpenai, SummaryBart, SummaryModel
from chatbot.global_state import GlobalState
from chatbot.emotion_manager import emotion_list

EMOTION_SUMMARY_FRACTION = 0.5
MODE_CALC_ALL = 1
MODE_CALC_CONTINUE = 2
MODE_CALC_LATEST = 3

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
        elif self.gs.config["summarizer"] == "model":
            summarizer = SummaryModel()
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
        self.calc_summaries(MODE_CALC_ALL, True)

    def calc_summaries(self, mode: int, clear: bool):
        if clear:
            self.cur.execute("delete from summaries")
            self.cur.execute("delete from summaries_messages")
            self.con.commit()

        res = self.cur.execute("SELECT * FROM characters")
        chars = res.fetchall()

        for i in range(len(chars)):
            char_id = chars[i]["id"]

            if mode == MODE_CALC_ALL:
                res = self.cur.execute("SELECT * FROM messages where character_id = ?", (char_id,))
                msgs = res.fetchall()
            elif mode == MODE_CALC_CONTINUE:
                res = self.cur.execute("SELECT max(last_message_id) as lmi FROM summaries where character_id = ?", (char_id,))
                tmp = res.fetchall()
                last_message_id = -10000000
                if len(tmp) > 0:
                    last_message_id = tmp[0]["lmi"]
                res = self.cur.execute("SELECT * FROM messages where character_id = ? and id > ?", (char_id, last_message_id))
                msgs = res.fetchall()
            elif mode == MODE_CALC_LATEST:
                lim = self.gs.config["summarizer_message_count"]
                sql = f"SELECT * from messages where id in (select id from messages where character_id = {char_id} order by id desc limit {lim}) order by id asc"
                res = self.cur.execute(sql)
                msgs = res.fetchall()

            for j in range(len(msgs)):
                cur_block_ids = []
                cur_block_msgs = []
                cur_emotions = {}

                emotion_counter = 0
                for emotion in emotion_list:
                    cur_emotions[emotion] = 0

                if j < self.gs.config["summarizer_message_count"]:
                    continue

                for k in range(self.gs.config["summarizer_message_count"]):
                    if j - k >= 0:
                        msg_id = msgs[j - k]["id"]
                        message = msgs[j - k]["character"] + ": " + msgs[j - k]["message"]
                        is_user = msgs[j - k]["is_user"]

                        cur_block_ids.insert(0, msg_id)
                        cur_block_msgs.insert(0, message)

                        if is_user == 0:
                            emotion_counter += 1
                            for emotion in emotion_list:
                                cur_emotions[emotion] = cur_emotions[emotion] + msgs[j - k][emotion]

                        if k == self.gs.config["summarizer_message_count"] - 1:
                            res = self.cur.execute("select * from summaries where last_message_id = ?",
                                                   (msg_id,))
                            sums = res.fetchall()
                            if len(sums) > 0:
                                cur_block_msgs.insert(0, sums[0]["summary"])

                                emotion_counter += 1
                                for emotion in emotion_list:
                                    cur_emotions[emotion] = cur_emotions[emotion] + (sums[0][emotion] * EMOTION_SUMMARY_FRACTION)
                    else:
                        break

                if len(cur_block_ids) == 0:
                    continue
                if len(cur_block_msgs) == 0:
                    continue

                whole_text = "\n".join(cur_block_msgs)
                whole_text.strip()

                summary = self.summarizer.summarize_text(whole_text)
                token_count = self.gs.model_manager.get_token_count(summary)

                send_date = datetime.now(timezone.utc)
                self.cur.execute(
                    "INSERT INTO summaries (character_id, original_text, summary, time, token_count, last_message_id) VALUES(?, ?, ?, ?, ?, ?)",
                    (char_id, whole_text, summary, send_date, token_count, cur_block_ids[-1]))
                self.con.commit()
                inserted_id = self.cur.lastrowid

                if emotion_counter > 0:
                    for emotion in emotion_list:
                        sql = f"update summaries set {emotion} = ? where id = ?"
                        self.cur.execute(sql, (cur_emotions[emotion] / emotion_counter, inserted_id))
                self.con.commit()

                # Insert relation to messages
                for msg_id in cur_block_ids:
                    self.cur.execute("INSERT INTO summaries_messages (message_id, summary_id) VALUES(?, ?)",
                                     (msg_id, inserted_id))
                    self.con.commit()
