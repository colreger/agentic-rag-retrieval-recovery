import json
import csv

# paths to the three important parts of the BEIR Natural Questions dataset

queries_path = "datasets/nq/queries.jsonl"
corpus_path = "datasets/nq/corpus.jsonl"
qrels_path = "datasets/nq/qrels/test.tsv"

# ---------------------------------------------------------
# 1. Load one query ID and its gold document from qrels.
#
# qrels is our "ground truth":
# It tells us which document is relevant for which question.
# ---------------------------------------------------------

with open(qrels_path, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file, delimiter="\t")

    first_row = next(reader)

    query_id = first_row["query-id"]
    gold_document_id = first_row["corpus-id"]

# ---------------------------------------------------------
# 2. Find the actual question belonging to this query ID.
# ---------------------------------------------------------

question = None

with open(queries_path,"r",encoding="utf-8") as file:
    for line in file:
        query = json.loads(line)

        if query["_id"] == query_id:
            question = query["text"]
            break


# ---------------------------------------------------------
# 3. Find the gold passage in the corpus.
#
# At this point we are NOT retrieving anything.
# We are simply looking up the passage that the dataset
# tells us is relevant.
# ---------------------------------------------------------

gold_document = None

with open(corpus_path, "r", encoding="utf-8") as file:
    for line in file:
        document = json.loads(line)

        if document["_id"] == gold_document_id:
            gold_document = document
            break

# print everything so that we can inspect

print(question)

print("\nGOLD DOCUMENT ID:")
print(gold_document_id)

print("\nGOLD PASSAGE:")
print(gold_document["text"])
