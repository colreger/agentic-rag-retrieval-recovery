# Agentic RAG Retrieval Recovery

This is a small project I built while exploring the idea for my Master thesis
about Agentic RAG.

I wanted to understand in practice what happens when retrieval fails, even
though the information needed to answer the question is actually in the corpus.

The basic idea is:

Question
→ Retrieval
→ Check if the retrieved information is enough
→ if not: reformulate the query
→ retrieve again


## What I did

I used Natural Questions from the BEIR benchmark and built a simple dense
retriever with Sentence Transformers (`all-MiniLM-L6-v2`).

The question and the documents are transformed into embeddings and I compare
them using cosine similarity. Then I retrieve the Top 5 documents.

First I searched for examples where the gold document was not in the Top 5.


## Something interesting I found

At first I thought:

gold document not in Top 5 = retrieval failed.

But this is actually not always true.

For example, for one question the gold document was missing, but another
document in the Top 5 contained basically the information needed to answer
the question.

So I had to distinguish between:

- gold document is missing
- the actual information/evidence needed for the answer is missing

For my experiment the second case is the interesting one.


## Example of a retrieval failure

One example I found was:

`royal society for the protection of birds number of members`

The correct passage exists in the corpus (`doc1042`), but it was not retrieved
in the initial Top 5.

The retrieved passages were about the RSPB, but did not contain the number
of members.

So the retrieval was semantically related to the question, but the evidence
was not enough to answer it.


## Recovery

First I tested manually if a better query could recover the missing document.

Changing the query was enough for the same retriever to find `doc1042`
at rank 1.

After that I added an LLM using the NVIDIA API.

The LLM gets:

- the original question
- the Top 5 retrieved passages

It does NOT get the gold document or the correct answer.

I ask it to decide:

`SUFFICIENT` or `INSUFFICIENT`

For the RSPB example it returned:

`INSUFFICIENT`

Then I asked the LLM to reformulate the query itself.

It generated:

`RSPB membership numbers 2023`

The next step is to use this new query for re-retrieval and check if the
missing evidence can be recovered.


## Why I am doing this

I am interested in the recovery process in Agentic RAG:

Where does recovery work and where does it fail?

For example:

1. Does the LLM notice that evidence is missing?
2. Can it create a better query?
3. Does the retriever find the missing evidence after that?
4. Can the system then answer the original question correctly?


## Tech

- Python
- BEIR / Natural Questions
- Sentence Transformers
- `all-MiniLM-L6-v2`
- NVIDIA API
- Nemotron