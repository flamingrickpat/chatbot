import chromadb
from sentence_transformers import SentenceTransformer
import sqlite3
from collections import OrderedDict

from scipy.signal import savgol_filter
import numpy as np

from chatbot.global_state import GlobalState
from chatbot.utils import np_to_blob, blob_to_np, cosine_sim, l2_squared

class ChromaManager:
    def __init__(self):
        self.gs = GlobalState()
        self.chroma = None
        self.col_messages = None
        self.col_summaries = None
        self.model = None

        self.initialize_chroma()

    def initialize_chroma(self):
        """
        Create client and collections.
        :return:
        """

        chroma_path = self.gs.config["chromadb_path"]
        embedder = self.gs.config["chromadb_embedder"]

        if embedder != "":
            chromadb_embedder = SentenceTransformer(embedder)
            self.model = chromadb_embedder

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

            distance = l2_squared(embedding_src, embedding)
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

    def get_related_messages(self, current_character_id, mem_ustm, mem_ltm_temp, max_token_count):
        class MsgItem:
            def __init__(self, id, priority, token_count):
                self.id = id
                self.priority = priority
                self.token_count = token_count

            def __eq__(self, other) -> bool:
                if isinstance(other, MsgItem):
                    return self.id == other.id
                return False

            def __hash__(self):
                return self.id

        mem_ltm = []
        token_count_ltm = 0

        chroma_dict = OrderedDict()
        tmp = []
        for id in mem_ustm:
            msg = self.gs.message_manager.get_message_per_id(id)
            vecs = self.get_results_db(is_message=True, character_id=current_character_id, text=msg, count=100)

            for i in range(len(vecs["ids"])):
                id = vecs["ids"][i]
                dist = vecs["distances"][i]
                tc = vecs["token_counts"][i]
                prio = 3 - dist

                if id in mem_ltm_temp and id not in tmp and prio >= 0:
                    item = MsgItem(id, prio, tc)
                    chroma_dict[id] = item
                    tmp.append(id)

        lst_tmp = []
        prios = []
        for key, value in chroma_dict.items():
            lst_tmp.append(value)
            prios.append(value.priority)

        yhat = savgol_filter(np.array(prios), self.gs.config["message_gauss_range"], 3)
        lst_smoothed = yhat.tolist()
        for i, val in enumerate(lst_smoothed):
            lst_tmp[i].priority = lst_smoothed[i]

        new_list = sorted(lst_tmp, key=lambda x: x.priority, reverse=True)
        for item in new_list:
            if token_count_ltm > max_token_count:
                break
            mem_ltm.append(item.id)
            token_count_ltm += item.token_count

        return mem_ltm