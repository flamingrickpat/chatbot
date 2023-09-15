import chromadb
from sentence_transformers import SentenceTransformer

from chatbot.global_state import GlobalState


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
