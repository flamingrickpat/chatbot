import sqlite3
import os

from chatbot.global_state import GlobalState
class DbManager:
    def __init__(self):
        self.gs = GlobalState()
        self.con = None
        self.cur = None
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