import json
import csv
from sentence_transformers import SentenceTransformer
import numpy as np


# ---------------------------------------------------------
# 1. File paths
# ---------------------------------------------------------

queries_path = "datasets/nq/queries.jsonl"
corpus_path = "datasets/nq/corpus.jsonl"
qrels_path = "datasets/nq/qrels/test.tsv"


# ---------------------------------------------------------
# 2. Read the first question and its gold document ID.
#
# We use the gold ID ONLY for evaluation later.
# The retriever itself will not receive this information.
# ---------------------------------------------------------

with open(qrels_path, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file, delimiter="\t")
    first_row = next(reader)

    query_id = first_row["query-id"]
    gold_document_id = first_row["corpus-id"]


# ---------------------------------------------------------
# 3. Find the question text.
# ---------------------------------------------------------

question = None

with open(queries_path, "r", encoding="utf-8") as file:
    for line in file:
        query = json.loads(line)

        if query["_id"] == query_id:
            question = query["text"]
            break


# ---------------------------------------------------------
# 4. Build a SMALL experimental corpus.
#
# For now, we load only 10,000 passages so the experiment
# runs quickly on a normal computer.
#
# IMPORTANT:
# We explicitly make sure that the gold passage is included.
# Therefore, if retrieval fails later, it is NOT because
# the gold passage was absent from the searchable corpus.
# ---------------------------------------------------------

documents = []
gold_document = None

with open(corpus_path, "r", encoding="utf-8") as file:
    for i, line in enumerate(file):
        document = json.loads(line)

        if i < 10000:
            documents.append(document)

        if document["_id"] == gold_document_id:
            gold_document = document

        if i >= 10000 and gold_document is not None:
            break


# Make absolutely sure the gold document is searchable.
if not any(doc["_id"] == gold_document_id for doc in documents):
    documents.append(gold_document)


print("Number of searchable passages:", len(documents))
print("Gold passage is in corpus:",
      any(doc["_id"] == gold_document_id for doc in documents))


# ---------------------------------------------------------
# 5. Load an embedding model.
#
# This model converts the question and every passage into
# numerical vectors (embeddings).
# ---------------------------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------------------------------------
# 6. Create embeddings for all passages.
#
# normalize_embeddings=True means that the dot product
# below is equivalent to cosine similarity.
# ---------------------------------------------------------

texts = [doc["text"] for doc in documents]

document_embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)

question_embedding = model.encode(
    question,
    normalize_embeddings=True
)


# ---------------------------------------------------------
# 7. Calculate similarity between the question and
# every passage.
# ---------------------------------------------------------

scores = document_embeddings @ question_embedding


# ---------------------------------------------------------
# 8. Sort passages from highest to lowest similarity
# and take the top 5.
# ---------------------------------------------------------

top_k = 5

top_indices = np.argsort(scores)[::-1][:top_k]


# ---------------------------------------------------------
# 9. Print the results.
# ---------------------------------------------------------

print("\nQUESTION:")
print(question)

print("\nGOLD DOCUMENT:")
print(gold_document_id)

print("\nTOP 5 RETRIEVED PASSAGES:\n")

retrieved_ids = []

for rank, index in enumerate(top_indices, start=1):

    document = documents[index]
    retrieved_ids.append(document["_id"])

    print("RANK", rank)
    print("Document ID:", document["_id"])
    print("Similarity:", round(float(scores[index]), 4))
    print(document["text"][:500])
    print("-" * 70)


# ---------------------------------------------------------
# 10. Evaluate retrieval.
#
# Only NOW do we use the gold document ID.
# ---------------------------------------------------------

if gold_document_id in retrieved_ids:
    print("\nRESULT: GOLD PASSAGE FOUND IN TOP-5 ✓")
else:
    print("\nRESULT: GOLD PASSAGE NOT FOUND IN TOP-5 ✗")

    print("\nTHE GOLD PASSAGE WAS:")
    print(gold_document["text"])