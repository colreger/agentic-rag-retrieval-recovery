import json
import csv
import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# 1. File paths
# ---------------------------------------------------------

queries_path = "datasets/nq/queries.jsonl"
corpus_path = "datasets/nq/corpus.jsonl"
qrels_path = "datasets/nq/qrels/test.tsv"


# ---------------------------------------------------------
# 2. Load the first 10,000 corpus passages.
#
# These passages will form our small experimental search
# space so that the test runs quickly on a normal computer.
# ---------------------------------------------------------

documents = []

with open(corpus_path, "r", encoding="utf-8") as file:
    for i, line in enumerate(file):

        if i >= 10000:
            break

        documents.append(json.loads(line))


# Create a dictionary so we can find documents by ID quickly.
document_by_id = {
    document["_id"]: document
    for document in documents
}


# ---------------------------------------------------------
# 3. Load all test questions.
# ---------------------------------------------------------

queries = {}

with open(queries_path, "r", encoding="utf-8") as file:
    for line in file:

        query = json.loads(line)

        queries[query["_id"]] = query["text"]


# ---------------------------------------------------------
# 4. Load gold document IDs from qrels.
# ---------------------------------------------------------

test_cases = []

with open(qrels_path, "r", encoding="utf-8") as file:

    reader = csv.DictReader(file, delimiter="\t")

    for row in reader:

        query_id = row["query-id"]
        gold_id = row["corpus-id"]

        # For this quick experiment we only use cases whose
        # gold passage is already inside our 10,000-document
        # search corpus.
        if gold_id in document_by_id:

            test_cases.append(
                (query_id, gold_id)
            )


print("Usable test questions:", len(test_cases))


# ---------------------------------------------------------
# 5. Load the same embedding model as before.
# ---------------------------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------------------------------------
# 6. Embed the 10,000 corpus passages once.
# ---------------------------------------------------------

texts = [
    document["text"]
    for document in documents
]

document_embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)


# ---------------------------------------------------------
# # ---------------------------------------------------------
# 7. Search for the first 10 cases where the annotated
# gold passage is NOT among the top 5 retrieved passages.
# ---------------------------------------------------------

number_of_failures = 0

for query_id, gold_id in test_cases:

    question = queries[query_id]

    question_embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    scores = document_embeddings @ question_embedding

    top_indices = np.argsort(scores)[::-1][:5]

    retrieved_ids = [
        documents[index]["_id"]
        for index in top_indices
    ]

    # We are interested only in cases where the annotated
    # gold passage was not retrieved.
    if gold_id not in retrieved_ids:

        number_of_failures += 1

        print("\n\n======================================")
        print("CASE", number_of_failures)
        print("GOLD PASSAGE NOT IN TOP-5")
        print("======================================")

        print("\nQUESTION:")
        print(question)

        print("\nGOLD DOCUMENT ID:")
        print(gold_id)

        print("\nGOLD PASSAGE:")
        print(document_by_id[gold_id]["text"])

        print("\nTOP 5 RETRIEVED INSTEAD:\n")

        for rank, index in enumerate(top_indices, start=1):

            document = documents[index]

            print("RANK", rank)
            print("Document ID:", document["_id"])
            print(
                "Similarity:",
                round(float(scores[index]), 4)
            )
            print(document["text"][:700])
            print("-" * 70)

        # Stop after ten examples.
        if number_of_failures == 10:
            break