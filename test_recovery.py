import json
import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# 1. Original question and a reformulated query.
#
# Later, the LLM agent will create this reformulation.
# For now, we write it manually so that we can test whether
# query reformulation can improve retrieval at all.
# ---------------------------------------------------------

original_question = (
    "royal society for the protection of birds number of members"
)

reformulated_query = (
    "RSPB membership numbers 2023"
)

gold_document_id = "doc1042"


# ---------------------------------------------------------
# 2. Load the same 10,000-document corpus as before.
# ---------------------------------------------------------

corpus_path = "datasets/nq/corpus.jsonl"

documents = []

with open(corpus_path, "r", encoding="utf-8") as file:
    for i, line in enumerate(file):

        if i >= 10000:
            break

        documents.append(json.loads(line))


# ---------------------------------------------------------
# 3. Load the same embedding model.
# ---------------------------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------------------------------------
# 4. Create embeddings for all documents.
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
# 5. Function for retrieving the top 5 passages.
# ---------------------------------------------------------

def retrieve(query):

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    scores = document_embeddings @ query_embedding

    top_indices = np.argsort(scores)[::-1][:5]

    return [
        (
            documents[index],
            float(scores[index])
        )
        for index in top_indices
    ]


# ---------------------------------------------------------
# 6. Run the ORIGINAL retrieval.
# ---------------------------------------------------------

original_results = retrieve(original_question)


print("\n====================================")
print("ORIGINAL QUERY")
print("====================================")

print(original_question)

for rank, (document, score) in enumerate(
    original_results,
    start=1
):
    print(
        rank,
        document["_id"],
        round(score, 4)
    )


# ---------------------------------------------------------
# 7. Run retrieval again with the REFORMULATED query.
# ---------------------------------------------------------

recovery_results = retrieve(reformulated_query)


print("\n====================================")
print("REFORMULATED QUERY")
print("====================================")

print(reformulated_query)

for rank, (document, score) in enumerate(
    recovery_results,
    start=1
):
    print(
        rank,
        document["_id"],
        round(score, 4)
    )


# ---------------------------------------------------------
# 8. Evaluate whether recovery succeeded.
# ---------------------------------------------------------

recovered_ids = [
    document["_id"]
    for document, score in recovery_results
]


if gold_document_id in recovered_ids:
    print("\nRECOVERY SUCCESS ✓")
    print("Gold passage is now in Top-5.")

else:
    print("\nRECOVERY FAILED ✗")
    print("Gold passage is still not in Top-5.")
