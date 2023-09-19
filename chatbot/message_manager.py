import sqlite3
import os
from typing import Dict, List, Any, Tuple
from datetime import datetime, timezone
import logging

import jellyfish

from chatbot.global_state import GlobalState
from chatbot.exceptions import *
from chatbot.utils import split_into_sentences

logger = logging.getLogger('message_manager')

str_input = "\n\n### Input:\n"
str_response = "\n\n### Response:\n"

class MessageManager():
    def __init__(self):
        self.gs = GlobalState()
        self.con = None
        self.cur = None

        self.current_character_id = 0
        self.current_character_name = ""

        self.init_database()

    def init_database(self) -> None:
        """
        Open database if it exists, otherwise create it.
        Same with chroma.
        """
        path = self.gs.config["database_path"]
        if os.path.exists(path):
            self.con = sqlite3.connect(path)
            self.con.row_factory = sqlite3.Row
            self.cur = self.con.cursor()
        else:
            self.create_database(path)

    def create_database(self, path: str) -> None:
        """
        Create database at location.
        """
        pass

    def list_available_characters(self) -> List[str]:
        """
        List all characters in database.
        """
        sql = "select name from characters"
        res = self.cur.execute(sql).fetchall()
        characters = []
        for char in res:
            characters.append(char["name"])
        return characters

    def add_character(self, name: str) -> None:
        """
        Add a new character.
        """
        sql = "select name from characters where name = ?"
        res = self.cur.execute(sql, (name,)).fetchall()
        if len(res) == 0:
            sql = "insert into characters (name) values (?)"
            self.cur.execute(sql, (name,))
            self.con.commit()
        else:
            raise CharacterAlreadyExistsException()

    def delete_character(self, name: str) -> None:
        """
        Add a new character.
        """
        sql = "select name from characters where name = ?"
        res = self.cur.execute(sql, (name,)).fetchall()
        if len(res) > 0:
            sql = "delete from characters where name = (?)"
            self.cur.execute(sql, (name,))
            self.con.commit()
        else:
            raise CharacterDoesntExistsException()

    def select_character(self, name: str) -> None:
        """
        Select a character.
        """
        sql = "select * from characters where name = ?"
        res = self.cur.execute(sql, (name,)).fetchall()
        if len(res) > 0:
            id = res[0]["id"]
            self.current_character_id = id
            self.current_character_name = res[0]["name"]

            self.gs.model_manager.reload_model(id)
        else:
            raise CharacterDoesntExistsException()

    def update_character_card(self, card: str) -> (str, str):
        """Update the character card of the current character."""
        model_manager = self.gs.model_manager
        token_count = model_manager.get_token_count(card)

        sql = "update characters set card = ?, token_count = ? where id = ?"
        self.cur.execute(sql, (card, token_count, self.current_character_id))
        self.con.commit()

        return self.current_character_name, token_count

    def insert_message(self, is_user: bool, message: str) -> int:
        """
        Insert message to database.
        Calculate token count.
        :param is_user:
        :param message:
        :return: inserted id
        """
        if is_user:
            character = self.gs.config["user_name"]
        else:
            character = self.current_character_name

        ratio = self.gs.emotion_manager.nsfw_ratio(message)
        full_message = character + ": " + message
        token_count = self.gs.model_manager.get_token_count(full_message) + 1
        send_date = datetime.now(timezone.utc)
        self.cur.execute(
            "INSERT INTO messages (character_id, is_user, character, message, time, token_count, nsfw_ratio) VALUES(?, ?, ?, ?, ?, ?, ?)",
            (self.current_character_id, is_user, character, message, send_date, token_count, ratio))
        self.con.commit()
        id = self.cur.lastrowid

        # insert into chroma db
        self.gs.chroma_manager.insert(is_message=True, id=id, character_id=self.current_character_id,
                                      is_user=is_user, text=full_message, token_count=token_count)

        # create new summary
        self.summarize_last_messages()

        return id

    def regenerate(self) -> (int, int, str):
        """
        Regenerate last message.
        Delete latest summary.
        First get the chat_id and message_id from the database, then delete the row.
        Call get_response to regenerate it and return it.
        :return: tuple with chat_id, message_id and new response
        """
        sql = "select * from summaries where character_id = ? order by id desc limit 1"
        res = self.cur.execute(sql, (self.current_character_id,)).fetchall()
        if len(res) > 0:
            id = res[0]["id"]
            sql = "delete from summaries where id = ?"
            self.cur.execute(sql, (id,))
            self.con.commit()

            self.gs.chroma_manager.delete(is_message=False, id=id)

        sql = "select * from messages where character_id = ? and is_user = 0 order by id desc limit 1"
        res = self.cur.execute(sql, (self.current_character_id,)).fetchall()
        if len(res) > 0:
            id = res[0]["id"]
            telegram_chat_id = res[0]["telegram_chat_id"]
            telegram_message_id = res[0]["telegram_message_id"]

            sql = "delete from messages where id = ?"
            self.cur.execute(sql, (id,))
            self.con.commit()

            self.gs.chroma_manager.delete(is_message=True, id=id)

            db_id, new_prompt = self.get_response()
            return db_id, telegram_chat_id, telegram_message_id, new_prompt
        else:
            raise CharacterDoesntExistsException()

    def clean_result(self, res: str) -> str:
        """
        Remove special tokens and names from result.
        """
        un = self.gs.config["user_name"]
        res = res.replace(f"{un}:", "")
        res = res.replace(f"{self.current_character_name}:", "")
        res = res.replace("</s>", "").strip()
        return res

    def check_similarity(self, messages: [str], res) -> bool:
        """
        Check jaro distance to other messages from model.
        :param messages:
        :param res:
        :return:
        """
        for m in messages:
            sim = jellyfish.jaro_distance(m, res)
            if sim > self.gs.config["max_jaro_distance"]:
                return False
        return True

    def check_similarity_sentences(self, messages: [str], res) -> bool:
        """
        Check jaro distance to other messages from model.
        :param messages:
        :param res:
        :return:
        """
        input_sentences = split_into_sentences(res)
        old_sentences = []
        for mes in messages:
            tmp = split_into_sentences(mes)
            for t in tmp:
                old_sentences.append(t)

        for inps in input_sentences:
            for olds in old_sentences:
                if len(inps) > 5 and len(olds) > 5:
                    sim = jellyfish.jaro_distance(inps, olds)
                    if sim > self.gs.config["max_jaro_distance"]:
                        return False
        return True

    def get_old_messages(self, limit: int) -> [str]:
        """
        Get old messages from model so that AI can't repeat itself within 20 messages.
        :return:
        """
        lst = []

        res = self.cur.execute("SELECT * FROM messages where character_id = ? order by id desc limit ?",
                               (self.current_character_id, limit))
        res = res.fetchall()
        for i in range(len(res) - 1, -1, -1):
            message = res[i]["message"]
            lst.append(message)

        return lst

    def get_response(self) -> (int, str):
        """
        Generate a prompt, send it to model and parse response. Save response in database.
        :return: tuple of id of new response in database and text
        """
        old_messages = self.get_old_messages(limit=10)

        prompt = self.get_prompt()

        prompt_path = self.gs.config["prompt_path"]
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write(prompt)

        tmp = prompt.encode('ascii', 'ignore').decode('ascii')
        logger.info("New prompt: " + tmp)

        text = ""
        db_id = 0

        cnt = 0
        while True:
            cnt += 1
            if cnt > 10:
                text = "Max tries, breaking loop."
                break
            try:
                text = ""

                # Try until character name is in response!
                self.gs.temperature_modifier = 0
                self.gs.top_p_modifier = 0
                while True:
                    text = self.call_model(prompt)
                    logger.info("New output: " + text.encode('ascii', 'ignore').decode('ascii'))
                    if self.check_similarity_sentences(old_messages, text):
                        text = self.clean_result(text)
                        break
                    else:
                        logger.info("Too similar, increasing temperature and top_p!")
                        self.gs.temperature_modifier = self.gs.temperature_modifier + \
                                                       self.gs.config["auto_raise_temperature"]
                        self.gs.top_p_modifier = self.gs.top_p_modifier + self.gs.config["auto_raise_top_p"]

                # Clean up and insert into db!
                if text != "":
                    # text = text.replace(f"{self.current_character_name}:", "").strip()
                    user_name = self.gs.config["user_name"]
                    text = text.replace(f"{user_name}:", "")
                    text = text.replace("### Input:", "")
                    text = text.replace("### Response:", "")
                    text = text.strip()
                    db_id = self.insert_message(is_user=False, message=text)
                    break
            except Exception as e:
                logger.error(str(e))

        return db_id, text

    def set_telegram_info(self, db_message_id: int, telegram_chat_id: int, telegram_message_id: int) -> None:
        """
        Update telegram info in db.
        :param db_message_id:
        :param telegram_chat_id:
        :param telegram_message_id:
        :return:
        """
        sql = "update messages set telegram_chat_id = ?, telegram_message_id = ? where id = ?"
        self.cur.execute(sql, (telegram_chat_id, telegram_message_id, db_message_id))
        self.con.commit()

    def get_prompt(self) -> str:
        """
        Generate current prompt with intro and messages to fit inside context size.
        """

        tokens_input = self.gs.model_manager.get_token_count(str_input)
        tokens_response = self.gs.model_manager.get_token_count(str_response)

        # Set current tokens to context size
        tokens_current = self.gs.config["context_size"]

        # Add character card to prompt
        res = self.cur.execute("SELECT * FROM characters where id = ?", (self.current_character_id,))
        res = res.fetchall()
        card = res[0]["card"].replace("\r", "")

        # Insert summaries to cards scenario section
        if "###SUMMARIES###" in card:
            summaries = self.get_relevant_summaries()
            card = card.replace("###SUMMARIES###", summaries)

        # Insert relevant messages to cards scenario section
        if "###MESSAGES###" in card:
            messages = self.get_relevant_messages()
            card = card.replace("###MESSAGES###", messages)
            raise Exception("Not supported anymore!")

        # Add card to prompt
        new_prompt = card

        # Recalculate token count to be safe
        token_count = self.gs.model_manager.get_token_count(new_prompt)
        tokens_current -= token_count

        # select all messages
        # add all messages to prompt so that enough space is left for relevant messages
        last_messages_ids = []
        messages_within_context = []
        res = self.cur.execute("SELECT * FROM messages where character_id = ?", (self.current_character_id,))
        res = res.fetchall()
        for i in range(len(res) - 1, -1, -1):
            id = res[i]["id"]
            is_user = res[i]["is_user"]
            message = res[i]["message"]
            character = res[i]["character"]
            token_count = res[i]["token_count"] + 1  # + 1 for newline

            msg = ""
            if self.gs.config["add_instructions"]:
                if is_user:
                    msg = msg + str_input
                    tokens_current -= tokens_input
                else:
                    msg = msg + str_response
                    tokens_current -= tokens_response
            msg = msg + character + ": " + message

            tokens_current -= token_count
            if tokens_current > self.gs.config["memory_message_length"]:
                messages_within_context.append(msg)
                last_messages_ids.append(id)
            else:
                break

        # make long string of last messages
        last_messages = ""
        for msg in reversed(messages_within_context):
            if self.gs.config["add_instructions"]:
                last_messages = last_messages + msg
            else:
                last_messages = last_messages + msg + "\n"


        # make long string of relevant messages that are NOT in last messages
        # Get last message of user
        allowance_relevant = self.gs.config["memory_message_length"]
        relevant_messages = ""
        sql = "select * from messages where character_id = ? and is_user = 1 order by id desc limit 1"
        res = self.cur.execute(sql, (self.current_character_id,))
        res = res.fetchall()
        if len(res) > 0:
            message = res[0]["message"]
            results = self.gs.chroma_manager.get_results(is_message=True, character_id=self.current_character_id,
                                                         text=message, count=100)
            for i in range(len(results['ids'][0])):
                id = results['metadatas'][0][i]["id"]
                text = results['documents'][0][i]
                tc = results['metadatas'][0][i]["token_count"]

                if (id not in last_messages_ids) and (text not in relevant_messages):
                    if allowance_relevant - tc > 0:
                        relevant_messages = relevant_messages + text + "\n"
                        allowance_relevant -= tc
                    else:
                        break

        # Combine prompt parts
        new_prompt = new_prompt + relevant_messages + last_messages

        # Add instruction and character name to prompt
        if self.gs.config["add_instructions"]:
            new_prompt = new_prompt + "\n\n### Response:\n" + f"{self.current_character_name}: "
        else:
            new_prompt = new_prompt.strip() + "\n" + f"{self.current_character_name}: "

        # Check final length
        final_length = self.gs.model_manager.get_token_count(new_prompt)
        logger.info("New prompt length: " + str(final_length))
        assert (final_length <= self.gs.config["context_size"])

        return new_prompt

    def call_model(self, prompt: str) -> str:
        """
        Append
        :param prompt:
        :return:
        """
        user_name = self.gs.config["user_name"]
        return self.gs.model_manager.get_message(prompt, stop_words=[f"{user_name}:", "\n"])

    def summarize_last_messages(self) -> None:
        """
        Call summarizer, summarize last x messages and write it to summaries table.
        """
        res = self.cur.execute("SELECT * FROM messages where character_id = ?", (self.current_character_id,))
        res = res.fetchall()

        text = ""

        in_window = res[max(0, len(res) - self.gs.config["summarizer_message_count"]):]
        ids = []
        for msg in in_window:
            ids.append(msg["id"])
            text = text + msg["character"] + ": " + msg["message"] + "\n"

        summary = self.gs.summarizer.summarize_text(text)
        token_count = self.gs.model_manager.get_token_count(summary)

        send_date = datetime.now(timezone.utc)
        self.cur.execute("INSERT INTO summaries (character_id, summary, time, token_count) VALUES(?, ?, ?, ?)",
                         (self.current_character_id, summary, send_date, token_count))
        self.con.commit()
        inserted_id = self.cur.lastrowid

        # insert into chroma
        self.gs.chroma_manager.insert(is_message=False, id=inserted_id, character_id=self.current_character_id,
                                      is_user=False, text=summary, token_count=token_count)

        # Insert relation to messages
        for msg_id in ids:
            self.cur.execute("INSERT INTO summaries_messages (message_id, summary_id) VALUES(?, ?)",
                             (msg_id, inserted_id))
            self.con.commit()

    def generate_missing_chroma_entries(self):
        self.gs.chroma_manager.clear(is_message=True)
        self.gs.chroma_manager.clear(is_message=False)

        self.gs.chroma_manager.initialize_chroma()

        res = self.cur.execute("SELECT * FROM messages")
        res = res.fetchall()

        for i in range(len(res)):
            id = res[i]["id"]
            character_id = res[i]["character_id"]
            is_user = res[i]["is_user"]
            message = res[i]["message"]
            character = res[i]["character"]
            token_count = res[i]["token_count"]

            self.gs.chroma_manager.insert(is_message=True, id=id, character_id=character_id,
                                          is_user=is_user, text=character + ": " + message,
                                          token_count=token_count)

        res = self.cur.execute("SELECT * FROM summaries")
        res = res.fetchall()
        for i in range(len(res)):
            id = res[i]["id"]
            character_id = res[i]["character_id"]
            is_user = False
            summary = res[i]["summary"]
            token_count = res[i]["token_count"]

            self.gs.chroma_manager.insert(is_message=False, id=id, character_id=character_id,
                                          is_user=is_user, text=summary,
                                          token_count=token_count)

    def get_relevant_summaries(self) -> str:
        """
        Get latest summary and load similar summaries from chroma to append to scenario.
        """
        max_token_length = self.gs.config["memory_summary_length"]

        sql = "select * from summaries where character_id = ? order by id desc limit 1"
        res = self.cur.execute(sql, (self.current_character_id,))
        res = res.fetchall()
        if len(res) > 0:
            summary = res[0]["summary"]

            results = self.gs.chroma_manager.get_results(is_message=False, character_id=self.current_character_id,
                                                         text=summary, count=100)
            result = self.gs.chroma_manager.cut_results_to_desired_length(results, max_token_length=max_token_length)
            return result.strip()
        else:
            return "Default scenario."

    def get_relevant_messages(self) -> str:
        """
        Get latest summary and load similar messages from chroma to append to scenario.
        """
        max_token_length = self.gs.config["memory_message_length"]

        sql = "select * from messages where character_id = ? and is_user = 1 order by id desc limit 1"
        res = self.cur.execute(sql, (self.current_character_id,))
        res = res.fetchall()
        if len(res) > 0:
            message = res[0]["message"]

            results = self.gs.chroma_manager.get_results(is_message=True, character_id=self.current_character_id,
                                                         text=message, count=100)
            result = self.gs.chroma_manager.cut_results_to_desired_length(results, max_token_length=max_token_length,
                                                                          add_newline=True)
            return result.strip()
        else:
            return "No relevant messages!"

    def get_messages_for_lora(self) -> str:
        """
        Get all messages of conversation with character.
        """
        messages_within_context = []
        new_prompt = ""

        res = self.cur.execute("SELECT * FROM messages where character_id = ?", (self.current_character_id,))
        res = res.fetchall()
        for i in range(len(res) - 1, -1, -1):
            is_user = res[i]["is_user"]
            message = res[i]["message"]
            character = res[i]["character"]

            msg = ""
            if is_user:
                msg = msg + str_input
            else:
                msg = msg + str_response
            msg = msg + character + ": " + message

            messages_within_context.append(msg)

        for msg in reversed(messages_within_context):
            new_prompt = new_prompt + msg

        return new_prompt

    def generate_missing_nsfw_ratio(self):
        """
        Go over all messages and find the nsfw ratio and update db.
        :return:
        """
        sql = "select * from messages"
        res = self.cur.execute(sql)
        res = res.fetchall()
        for r in res:
            id = r["id"]
            msg = r["message"]
            ratio = self.gs.emotion_manager.nsfw_ratio(msg)
            sql = "update messages set nsfw_ratio = ? where id = ?"
            self.cur.execute(sql, (ratio, id))
            self.con.commit()


