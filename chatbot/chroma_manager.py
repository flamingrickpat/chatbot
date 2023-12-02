import chromadb
from sentence_transformers import SentenceTransformer
import sqlite3

from chatbot.global_state import GlobalState
from chatbot.utils import np_to_blob, blob_to_np, cosine_sim

class ChromaManager:
    def __init__(self):
        self.gs = GlobalState()
        self.chroma = None
        self.col_messages = None
        self.col_summaries = None

        self.initialize_chroma()

    def initialize_chroma(self):
        """
        Create client and collections.
        :return:
        """

        chroma_path = self.gs.config["chromadb_path"]
        embedder = self.gs.config["chromadb_embedder"]
        self.chroma = chromadb.PersistentClient(path=chroma_path)

        if embedder != "":
            chromadb_embedder = SentenceTransformer(embedder)
            self.model = chromadb_embedder
            emb_fn = lambda *args, **kwargs: chromadb_embedder.encode(*args, **kwargs).tolist()

            self.col_messages = self.chroma.get_or_create_collection(name="messages", embedding_function=emb_fn)
            self.col_summaries = self.chroma.get_or_create_collection(name="summaries", embedding_function=emb_fn)
        else:
            self.col_messages = self.chroma.get_or_create_collection(name="messages")
            self.col_summaries = self.chroma.get_or_create_collection(name="summaries")

    def insert(self, is_message: bool, id: int, character_id: int, is_user: bool, text: str, token_count: int) -> None:
        if is_message:
            collection = self.col_messages
        else:
            collection = self.col_summaries

        collection.add(
            documents=[text],
            metadatas=[{"id": id, "source": "database", "is_user": is_user, "token_count": token_count,
                        "character_id": character_id}],
            ids=[f"{id}"]
        )

    def delete(self, is_message: bool, id: int) -> None:
        if is_message:
            collection = self.col_messages
        else:
            collection = self.col_summaries

        collection.delete(
            ids=[f"{id}"]
        )

    def clear(self, is_message: bool) -> None:
        if is_message:
            self.chroma.delete_collection("messages")
        else:
            self.chroma.delete_collection("summaries")


    def get_results(self, is_message: bool, character_id: int, text: str, count: int) -> str:
        if is_message:
            collection = self.col_messages
        else:
            collection = self.col_summaries

        results = collection.query(
            query_texts=[text],
            n_results=count,
            where={"character_id": character_id},
        )
        return results

    def cut_results_to_desired_length(self, results, max_token_length: int, add_newline: bool = False) -> str:
        res = ""
        tmp = max_token_length
        for i in range(len(results['ids'][0])):
            text = results['documents'][0][i]
            tc = results['metadatas'][0][i]["token_count"]

            if text in res:
                continue
            else:
                if (tmp - tc) > 0:
                    tmp -= tc
                    if add_newline:
                        res = res + text + "\n"
                    else:
                        res = res + text + " "
                else:
                    break

        return res


    def calc_embeddings_messages(self, id):
        con = self.gs.db_manager.con
        cur = self.gs.db_manager.cur

        if id is None:
            sql = "select * from messages"
        else:
            sql = f"select * from messages where id = {id}"
        res = cur.execute(sql)
        res = res.fetchall()
        for r in res:
            id = r["id"]
            msg = r["message"]

            embeddings = self.model.encode(msg)
            blob = np_to_blob(embeddings)

            sql = "update messages set embedding = ? where id = ?"
            cur.execute(sql, (sqlite3.Binary(blob), id))
        con.commit()

    def calc_embeddings_summaries(self, id):
        con = self.gs.db_manager.con
        cur = self.gs.db_manager.cur

        if id is None:
            sql = "select * from summaries"
        else:
            sql = f"select * from summaries where id = {id}"
        res = cur.execute(sql)
        res = res.fetchall()
        for r in res:
            id = r["id"]
            summary = r["summary"]

            embeddings = self.model.encode(summary)
            blob = np_to_blob(embeddings)

            sql = "update summaries set embedding = ? where id = ?"
            cur.execute(sql, (sqlite3.Binary(blob), id))
        con.commit()

    def text_to_embedding_blob(self, text):
        embeddings = self.model.encode(text)
        blob = np_to_blob(embeddings)
        return blob

    def get_results_db(self, is_message: bool, character_id: int, text: str, count: int) -> dict:
        con = self.gs.db_manager.con
        cur = self.gs.db_manager.cur

        embedding_src = self.model.encode(text)

        if is_message:
            res = cur.execute("select * from messages where character_id = ?", (character_id,)).fetchall()
        else:
            res = cur.execute("select * from summaries where character_id = ?", (character_id,)).fetchall()

        class Item():
            def __init__(self, id, distance, token_count):
                self.id = id
                self.distance = distance
                self.token_count = token_count

        items = []
        for i, row in enumerate(res):
            id = row["id"]
            blob = row["embedding"]
            token_count = row["token_count"]
            embedding = blob_to_np(blob)

            distance = cosine_sim(embedding_src, embedding)
            items.append(Item(id, distance, token_count))

        new_list = sorted(items, key=lambda x: x.distance, reverse=False)
        res = {
            "ids": [],
            "distances": [],
            "token_counts": [],
        }
        for i in range(count):
            if i >= len(new_list):
                break
            res["ids"].append(new_list[i].id)
            res["distances"].append(new_list[i].distance)
            res["token_counts"].append(new_list[i].token_count)

        return res
