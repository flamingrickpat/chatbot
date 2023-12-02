from enum import Enum
import re
import contextlib
import io
import sqlite3

import stanza
from nltk.stem import PorterStemmer

from chatbot.global_state import GlobalState
"""
Most of this code is stolen from: https://github.com/cisco-open/DeepVision/tree/main/recallm
"""

TEXT_CHUNK_SIZE = 2500
TEXT_CHUNK_OVERLAP = 1000

CONTEXT_REVISION_PERIOD = 3  # Number of iterations before context is revised
CONTEXT_SUMMARIZATION_DESIRED_SIZE = 80  # Size in number of words (Only for OpenAI chain)

NEIGHBORING_CONCEPT_DISTANCE_FOR_KNOWLEDGE_UPDATE = 1  # Number of related concepts to fetch -> Warning this causes exponential growth

class ConceptType(Enum):
    MERGED = 1
    MULTIPLE_PER_SENTENCE = 2


class Concept:
    def __init__(self, name, type, start_index, end_index, chunk_index) -> None:
        name = re.sub(r'[^\w\s]', '', name)  # Remove special symbols
        name = re.sub(r'\s+', '', name)  # Remove whitespace

        self.name = name
        self.type = type

        # These variables need to be overwritten later
        self.context = ""
        self.related_concepts = []
        self.unique_id = 0  # This id is only unique to the source text that the concept was extracted from. It is not unique in the entire concept knowledge base.
        self.chroma_id = name  # This id is unique in the chroma database.

        # These variables are only relevant when first extracting context from source and adding to databases
        self.start_index = start_index  # Refers to position in source text
        self.end_index = end_index
        self.chunk_index = chunk_index

        self.MERGES_BEFORE_REVISION = CONTEXT_REVISION_PERIOD
        self.merge_count = 0
        self.revision_count = 0  # Number of times that context has been updated, context is revised every n updates

        # Variables for when questioning the system
        self.t_index = 0  # t_index is only set and used when reconstructing the graph for graph traversal in fetch_contexts_for_question

        self.sort_val = 0

    def __lt__(self, other):
        return self.start_index < other.start_index

    def __eq__(self, other) -> bool:
        if isinstance(other, Concept):
            return self.name == other.name
        return False

    # Merge two different concepts to create one concept
    #   when full_merge is true -> merge contexts and related concepts as well
    def merge_with(self, concept, full_merge=False):
        if self.end_index <= concept.start_index:
            if self.name != concept.name:
                self.name = f'{self.name}{concept.name}'
            self.end_index = concept.end_index
        else:
            if self.name != concept.name:
                self.name = f'{concept.name}{self.name}'
            self.start_index = concept.start_index

        if self.type != concept.type:
            if self.type[0] < concept.type[0]:  # Concept is just a string so we want to make sure we always concate in alphabetical order so we don't get duplicates
                self.type = f'{self.type}|{concept.type}'
            else:
                self.type = f'{concept.type}|{self.type}'

        if full_merge:
            self.merge_count += 1

            self.context = f'{self.context} {concept.context}'

            for related_concept in concept.related_concepts:
                if related_concept not in self.related_concepts:
                    self.related_concepts.append(related_concept)

    def should_revise(self) -> bool:
        if self.revision_count > 0 and self.revision_count % CONTEXT_REVISION_PERIOD == 0:
            return True

        if self.merge_count >= self.MERGES_BEFORE_REVISION:
            return True

        return False

    def revise_context(self, summarization_chain):
        self.context = summarization_chain.summarize(self.context)
        self.merge_count = 0

    def __repr__(self) -> str:
        # return f'{self.name}:{self.type}\t{self.start_index}'
        return f'{self.name}:\t{self.context}'
        # return f'{self.name}: \t{[c.name for c in self.related_concepts]}'


def fetch_contexts_for_concepts(concepts, texts, context_size):
    if len(texts) == 0:
        return
    if type(texts) == str:  # There was no batching in texts
        texts = [texts]  # So we put it in batch format to work with rest of code

    for concept in concepts:
        current_text = texts[concept.chunk_index]
        concept.context = current_text[
                          max(0, concept.start_index - context_size):
                          min(len(current_text), concept.end_index + context_size)]
        concept.context = re.sub(r'[^\w\s]', '', concept.context)  # Remove special symbols


# Given a list of concepts, find the neighbouring concepts for all concepts
# by relative position in source text
def fetch_neigbouring_concepts(concepts, distance):
    concepts = sorted(concepts)
    for i in range(len(concepts)):
        concepts[i].related_concepts = []
        # concepts[i].chroma_id = ""
        for j in range(-distance, distance + 1, 1):  # If distance from current concept is less than parameter distance
            if i + j >= 0 and i + j < len(concepts):  # If index is in bounds
                # concepts[i].chroma_id = f'{concepts[i].chroma_id} {concepts[i+j].name}'

                if j == 0:
                    continue

                if concepts[i].name < concepts[
                    i + j].name:  # Ensure that we only create one connection between nodes in Neo4J graph
                    concepts[i].related_concepts.append(concepts[i + j])


# Given a list of concepts, merge all concepts that share the same name (Regardless
#   of their location in the source text)
def merge_concepts(concepts):
    result_concepts = []
    for i in range(len(concepts)):
        if concepts[i] not in result_concepts:
            result_concepts.append(concepts[i])

        for j in range(i + 1, len(concepts), 1):
            if concepts[i].name == concepts[j].name:
                concepts[i].merge_with(concept=concepts[j],
                                       full_merge=True)

    return result_concepts


def assign_unique_ids_to_concepts(concepts, chunk_index):
    for (i, concept) in enumerate(concepts):
        concept.unique_id = f'c{chunk_index}{i}'


class ConceptManager():
    def __init__(self):
        self.nltk_pipeline = self.create_nltk_pipeline()
        self.word_stemmer = PorterStemmer()
        self.gs = GlobalState()

    def create_nltk_pipeline(self):
        with contextlib.redirect_stdout(io.StringIO()):  # Supress console output
            stanza.download('en', verbose=False)
            nlp = stanza.Pipeline('en', verbose=False)

        return nlp
    def fetch_concepts_from_single_batch(self,
                                         text,
                                         batch_index,
                                         concept_type):
        doc = self.nltk_pipeline(text)

        concepts = []
        for sentence in doc.sentences:
            sentence_concepts = {}
            sentence_concept_string = ""

            index = 0
            for word in sentence.words:
                if word.upos in ["NOUN", "PROPN", "NUM"]:
                    word_concept_string = self.word_stemmer.stem(word.text).lower()
                    sentence_concepts[word_concept_string] = {'pos': word.upos, 'index': index}
                    index += 1

                    if concept_type == ConceptType.MERGED:
                        sentence_concept_string = f'{sentence_concept_string} {word_concept_string}'

            if len(sentence_concepts.keys()) > 3:  # Number of concepts in sentence is too much, we need to reduce it
                # If it contains proper noun -> Reduce to proper noun and surrounding nouns
                if "PROPN" in [value['pos'] for value in sentence_concepts.values()]:
                    propn_index = [element['index'] for element in sentence_concepts.values() if
                                   element['pos'] == "PROPN"]
                    propn_index = propn_index[0]

                    sentence_concepts_new = []
                    for text, meta in zip(sentence_concepts.keys(), sentence_concepts.values()):
                        if abs(meta['index'] - propn_index) <= 1:
                            sentence_concepts_new.append(text)

                    if concept_type == ConceptType.MERGED:
                        sentence_concept_string = ""
                        for concept in sentence_concepts_new:
                            sentence_concept_string = f'{sentence_concept_string} {concept}'
                # Else just use middle nouns
                else:
                    sentence_concepts_new = []
                    for text, meta in zip(sentence_concepts.keys(), sentence_concepts.values()):
                        middle_index = len(sentence_concepts.keys()) // 2
                        if abs(meta['index'] - middle_index) <= 1:
                            sentence_concepts_new.append(text)

                    if concept_type == ConceptType.MERGED:
                        sentence_concept_string = ""
                        for concept in sentence_concepts_new:
                            sentence_concept_string = f'{sentence_concept_string} {concept}'

                # Update sentence_concepts dictionary
                keys = list(sentence_concepts.keys())
                for key in keys:
                    if key not in sentence_concepts_new:
                        del sentence_concepts[key]

            if len(sentence_concepts.values()) == 0:
                continue

            # Get original sentence string back from [stanfordnlp.pipeline.word]
            sentence_context = sentence.words[0].text
            for word in sentence.words[1:]:
                if word.upos == "PUNCT":
                    sentence_context = f'{sentence_context}{word.text}'
                else:
                    sentence_context = f'{sentence_context} {word.text}'

            if concept_type == ConceptType.MULTIPLE_PER_SENTENCE:
                for concept_key, concept_value in zip(sentence_concepts.keys(), sentence_concepts.values()):
                    concept_index = concept_value['index'] + (batch_index * TEXT_CHUNK_SIZE)
                    sentence_concept_string = sentence_concept_string[1:]
                    new_concept = Concept(
                        name=concept_key,
                        type=concept_value['pos'],
                        start_index=concept_index,
                        end_index=concept_index,
                        chunk_index=batch_index
                    )
                    new_concept.context = sentence_context

                    concepts.append(new_concept)

            if concept_type == ConceptType.MERGED:
                concept_index = list(sentence_concepts.values())[0]['index'] + (batch_index * TEXT_CHUNK_SIZE)
                sentence_concept_string = sentence_concept_string[1:]
                new_concept = Concept(
                    name=sentence_concept_string,
                    type="",
                    start_index=concept_index,
                    end_index=concept_index,
                    chunk_index=batch_index
                )
                new_concept.context = sentence_context

                concepts.append(new_concept)

        return concepts

    # Root function to fetch concepts from a source text
    def fetch_concepts_from_texts(self, texts):
        if len(texts) == 0:
            return []
        if type(texts) != list:  # There was no batching in texts input
            texts = [texts]  # So we put it in batch format to work with rest of code

        concepts_chunks = []  # Array of chuhks/batches
        for (i, text) in enumerate(texts):
            new_concepts = self.fetch_concepts_from_single_batch(text=text,
                                                            batch_index=i,
                                                            concept_type=ConceptType.MULTIPLE_PER_SENTENCE)

            concepts_chunks.append(new_concepts)

        for i in range(len(concepts_chunks)):
            assign_unique_ids_to_concepts(
                concepts=concepts_chunks[i],
                chunk_index=i
            )

            fetch_neigbouring_concepts(
                concepts=concepts_chunks[i],
                distance=NEIGHBORING_CONCEPT_DISTANCE_FOR_KNOWLEDGE_UPDATE
            )

            concepts_chunks[i] = merge_concepts(concepts=concepts_chunks[i])

        return concepts_chunks

    def calc_concepts_summaries(self, id):
        con = self.gs.db_manager.con
        cur = self.gs.db_manager.cur

        def get_concept_id(name):
            res1 = cur.execute("select * from graph_nodes where name = ?", (name,)).fetchall()
            return res1[0]["id"]

        def upsert_concept(name):
            res1 = cur.execute("select * from graph_nodes where name = ?", (name,)).fetchall()
            if len(res1) > 0:
                concept_id = res1[0]["id"]
                revision_count = res1[0]["revision_count"] + 1
                cur.execute("update graph_nodes set revision_count = ? where id = ?", (revision_count, concept_id))
            else:
                cur.execute("insert into graph_nodes (name, revision_count) values (?, ?)", (name, 1))

        def upsert_relation(name_src, name_dest):
            src = get_concept_id(name_src)
            dest = get_concept_id(name_dest)

            res1 = cur.execute("select * from graph_relations where src_node_id = ? and dest_node_id = ?", (src, dest)).fetchall()
            if len(res1) > 0:
                rel_id = res1[0]["id"]
                revision_count = res1[0]["revision_count"] + 1
                cur.execute("update graph_relations set revision_count = ? where id = ?", (revision_count, rel_id))
            else:
                cur.execute("insert into graph_relations (src_node_id, dest_node_id, revision_count) values (?, ?, ?)", (src, dest, 1))

        def insert_context(sum_id, concept_name, concept_context, context_embedding, tc):
            node = get_concept_id(concept_name)
            cur.execute("insert into graph_context (node_id, summary_id, context, token_count, embedding) values (?, ?, ?, ?, ?)",
                        (node, sum_id, concept_context, tc, sqlite3.Binary(context_embedding)))

        if id is None:
            sql = "select * from summaries"
        else:
            sql = f"select * from summaries where id = {id}"
        res = cur.execute(sql)
        res = res.fetchall()
        for r in res:
            summary_id = r["id"]
            summary = r["summary"]

            concepts = self.fetch_concepts_from_texts([summary])

            for chunk in concepts:
                for mini_batch_i, concept in enumerate(chunk):
                    upsert_concept(concept.name)
                    blob = self.gs.chroma_manager.text_to_embedding_blob(concept.context)
                    token_count = self.gs.model_manager.get_token_count(concept.context)
                    insert_context(summary_id, concept.name, concept.context, blob, token_count)

            for chunk in concepts:
                for mini_batch_i, concept in enumerate(chunk):
                    for rel_concept in concept.related_concepts:
                        upsert_relation(concept.name, rel_concept.name)

            con.commit()
