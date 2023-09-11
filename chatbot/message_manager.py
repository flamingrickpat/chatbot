import sqlite3
import os
from typing import Dict, List, Any, Tuple
from datetime import datetime, timezone
import logging

from chatbot.global_state import GlobalState
from chatbot.exceptions import *

logger = logging.getLogger('message_manager')

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

        token_count = self.gs.model_manager.get_token_count(character + ": " + message)
        send_date = datetime.now(timezone.utc)
        self.cur.execute("INSERT INTO messages (character_id, is_user, character, message, time, token_count) VALUES(?, ?, ?, ?, ?, ?)",
                    (self.current_character_id, is_user, character, message, send_date, token_count))
        self.con.commit()

        return self.cur.lastrowid

    def get_response(self) -> (int, str):
        """
        Generate a prompt, send it to model and parse response. Save response in database.
        :return: tuple of id of new response in database and text
        """
        prompt = self.get_prompt()
        logger.info(prompt.replace("\n", ""))

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
                while True:
                    text = self.call_model(prompt)
                    if f"{self.current_character_name}:" in text:
                        break

                # Clean up and insert into db!
                if text != "":
                    text = text.replace(f"{self.current_character_name}:", "").strip()
                    user_name = self.gs.config["user_name"]
                    text = text.replace(f"{user_name}:", "").strip()
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

        # Set current tokens to context size
        tokens_current = self.gs.config["context_size"]
        new_prompt = ""

        # Add character card to prompt
        res = self.cur.execute("SELECT * FROM characters where id = ?", (self.current_character_id,))
        res = res.fetchall()
        card = res[0]["card"]
        token_count = res[0]["token_count"]
        tokens_current -= token_count

        new_prompt = new_prompt + card + "\n"

        # Add messages to prompt
        messages_within_context = []

        # select all messages
        res = self.cur.execute("SELECT * FROM messages where character_id = ?", (self.current_character_id,))
        res = res.fetchall()
        for i in range(len(res) - 1, -1, -1):
            message = res[i]["message"].strip()
            character = res[i]["character"]
            token_count = res[i]["token_count"] + 1  # + 1 for newline
            msg = character + ": " + message

            tokens_current -= token_count
            if tokens_current > 0:
                messages_within_context.append(msg)
            else:
                break

        for msg in reversed(messages_within_context):
            new_prompt = new_prompt + msg.strip() + "\n"

        # Check final length
        final_length = self.gs.model_manager.get_token_count(new_prompt)
        assert(final_length <= self.gs.config["context_size"])

        return new_prompt

    def call_model(self, prompt: str) -> str:
        return "Test: this is a test response!"