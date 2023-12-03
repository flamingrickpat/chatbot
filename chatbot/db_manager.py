import sqlite3
import os

from chatbot.global_state import GlobalState

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS "summaries_messages" (
	"id"	INTEGER,
	"message_id" integer,
	"summary_id" integer,
	PRIMARY KEY("id" AUTOINCREMENT)
);
CREATE TABLE IF NOT EXISTS "characters" (
	"id"	INTEGER,
	"name"	TEXT,
	"card"	TEXT,
	"token_count"	INTEGER,
	PRIMARY KEY("id" AUTOINCREMENT)
);
CREATE TABLE IF NOT EXISTS "finetuned_models" (
	"id"	INTEGER NOT NULL,
	"character_id"	INTEGER,
	"base_model"	TEXT,
	"path"	TEXT,
	"time"	TEXT,
	PRIMARY KEY("id" AUTOINCREMENT)
);
CREATE TABLE IF NOT EXISTS "messages" (
	"id"	INTEGER,
	"character_id"	INTEGER,
	"is_user"	INTEGER,
	"character"	INTEGER,
	"message"	TEXT,
	"time"	TEXT,
	"token_count"	INTEGER,
	"telegram_chat_id"	INTEGER,
	"telegram_message_id"	INTEGER,
	"nsfw_ratio"	NUMERIC,
	"caring"	REAL,
	"faithful"	REAL,
	"content"	REAL,
	"sentimental"	REAL,
	"joyful"	REAL,
	"hopeful"	REAL,
	"proud"	REAL,
	"guilty"	REAL,
	"sad"	REAL,
	"grateful"	REAL,
	"afraid"	REAL,
	"ashamed"	REAL,
	"trusting"	REAL,
	"confident"	REAL,
	"prepared"	REAL,
	"anxious"	REAL,
	"lonely"	REAL,
	"terrified"	REAL,
	"devastated"	REAL,
	"disgusted"	REAL,
	"annoyed"	REAL,
	"excited"	REAL,
	"jealous"	REAL,
	"anticipating"	REAL,
	"furious"	REAL,
	"angry"	REAL,
	"impressed"	REAL,
	"nostalgic"	REAL,
	"surprised"	REAL,
	"apprehensive"	REAL,
	"disappointed"	REAL,
	"embarrassed"	REAL,
	"embedding"	BLOB,
	PRIMARY KEY("id" AUTOINCREMENT)
);
CREATE TABLE IF NOT EXISTS "summaries" (
	"id"	INTEGER,
	"character_id"	INTEGER,
	"original_text"	TEXT,
	"summary"	TEXT,
	"time"	TEXT,
	"token_count"	INTEGER,
	"last_message_id"	INTEGER,
	"caring"	REAL,
	"faithful"	REAL,
	"content"	REAL,
	"sentimental"	REAL,
	"joyful"	REAL,
	"hopeful"	REAL,
	"proud"	REAL,
	"guilty"	REAL,
	"sad"	REAL,
	"grateful"	REAL,
	"afraid"	REAL,
	"ashamed"	REAL,
	"trusting"	REAL,
	"confident"	REAL,
	"prepared"	REAL,
	"anxious"	REAL,
	"lonely"	REAL,
	"terrified"	REAL,
	"devastated"	REAL,
	"disgusted"	REAL,
	"annoyed"	REAL,
	"excited"	REAL,
	"jealous"	REAL,
	"anticipating"	REAL,
	"furious"	REAL,
	"angry"	REAL,
	"impressed"	REAL,
	"nostalgic"	REAL,
	"surprised"	REAL,
	"apprehensive"	REAL,
	"disappointed"	REAL,
	"embarrassed"	REAL,
	"embedding"	BLOB,
	PRIMARY KEY("id" AUTOINCREMENT)
);
CREATE TABLE IF NOT EXISTS "graph_nodes" (
	"id"	INTEGER NOT NULL,
	"name"	TEXT,
	"revision_count"	INTEGER,
	"temporal_index"	INTEGER,
	PRIMARY KEY("id" AUTOINCREMENT)
);
CREATE TABLE IF NOT EXISTS "graph_relations" (
	"id"	INTEGER NOT NULL,
	"src_node_id"	INTEGER,
	"dest_node_id"	INTEGER,
	"revision_count"	INTEGER,
	PRIMARY KEY("id" AUTOINCREMENT)
);
CREATE TABLE IF NOT EXISTS "graph_context" (
	"id"	INTEGER NOT NULL,
	"node_id"	INTEGER,
	"summary_id"	INTEGER,
	"context"	TEXT,
	"token_count"	INTEGER,
	"embedding"	BLOB,
	PRIMARY KEY("id" AUTOINCREMENT)
);
"""

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
            self.con = sqlite3.connect(path, check_same_thread=False)
            self.con.row_factory = sqlite3.Row
            self.cur = self.con.cursor()
        else:
            self.con = sqlite3.connect(path, check_same_thread=False)
            self.con.row_factory = sqlite3.Row
            self.cur = self.con.cursor()
            self.cur.executescript(DB_SCHEMA)
            self.con.commit()

    def create_database(self, path: str) -> None:
        """
        Create database at location.
        """
        pass