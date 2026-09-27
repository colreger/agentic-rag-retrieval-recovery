import os
from dotenv import load_dotenv
from openai import OpenAI


# ---------------------------------------------------------
# 1. Load the NVIDIA API key from the .env file.
# ---------------------------------------------------------

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")


# ---------------------------------------------------------
# 2. Create a client for the NVIDIA API.
# ---------------------------------------------------------

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)


# ---------------------------------------------------------
# 3. This is the original question from our experiment.
# ---------------------------------------------------------

question = (
    "royal society for the protection of birds number of members"
)


# ---------------------------------------------------------
# 4. These are the five passages returned by our FIRST
# retrieval.
#
# Notice: They are relevant to the RSPB, but none of them
# contains the requested number of members.
# ---------------------------------------------------------

retrieved_passages = """
1. The Royal Society for the Protection of Birds (RSPB) is a
charitable organisation registered in England and Wales and
in Scotland. It was founded as the Plumage League in 1889.

2. The Plumage League was founded in 1889 by Emily Williamson
as a protest group campaigning against the use of bird feathers
in clothing.

3. The Society attracted support from influential figures and
received a Royal Charter in 1904.

4. The RSPB Medal is the Society's most prestigious award.

5. Today, the RSPB works with the civil service and Government
on conservation policy.
"""


# ---------------------------------------------------------
# 5. Ask the LLM to evaluate whether the retrieved evidence
# is sufficient.
#
# IMPORTANT:
# The model is not told the correct answer.
# It only sees the question and the retrieved passages.
# ---------------------------------------------------------

prompt = f"""
Question:

{question}

Retrieved passages:

{retrieved_passages}

Determine whether the retrieved passages contain enough
information to answer the question reliably.

Respond with exactly one word:

SUFFICIENT

or

INSUFFICIENT
"""


# ---------------------------------------------------------
# 6. Send the evaluation request to the LLM.
# ---------------------------------------------------------

response = client.chat.completions.create(
	model="nvidia/nemotron-3-super-120b-a12b",
    	messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    temperature=0
)


# ---------------------------------------------------------
# 7. Print the model's decision.
# ---------------------------------------------------------

decision = response.choices[0].message.content.strip()

print("\nQUESTION:")
print(question)

print("\nLLM DECISION:")
print(decision)

# ---------------------------------------------------------
# 8. If the retrieved information is insufficient,
# ask the same LLM to create a better retrieval query.
# ---------------------------------------------------------

if decision == "INSUFFICIENT":

    reformulation_prompt = f"""
The following search query did not retrieve enough information
to answer the question.

Original question:
{question}

Retrieved passages:
{retrieved_passages}

Create a better search query that is more likely to retrieve
the missing information needed to answer the original question.

Return ONLY the new search query.
Do not explain your answer.
"""

    reformulation_response = client.chat.completions.create(
        model="nvidia/nemotron-3-super-120b-a12b",
        messages=[
            {
                "role": "user",
                "content": reformulation_prompt
            }
        ],
        temperature=0
    )

    new_query = (
        reformulation_response
        .choices[0]
        .message
        .content
        .strip()
    )

    print("\nREFORMULATED QUERY:")
    print(new_query)