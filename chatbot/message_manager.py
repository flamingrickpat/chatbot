import sqlite3
import os
from typing import Dict, List, Any, Tuple

from chatbot.global_state import GlobalState
from chatbot.exceptions import *

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
        #model_manager = self.gs.model_manager
        token_count = 100 #model_manager.get_token_count(card)

        sql = "update characters set card = ?, token_count = ? where id = ?"
        self.cur.execute(sql, (card, token_count, self.current_character_id))
        self.con.commit()

        return self.current_character_name, token_count