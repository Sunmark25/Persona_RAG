import os

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed


WORKING_DIR = "./dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete,
    embedding_func=openai_embed,
    # llm_model_func=gpt_4o_complete
)


with open("./dickens/book.txt", "r", encoding="utf-8") as f:
    rag.insert(f.read())


empty_history = []


# # Perform local search
# print(
#     rag.query("As Scoorge, what do you think of Christmas?", param=QueryParam(mode="local", conversation_history=empty_history))
# )

# # Perform global search
# print(
#     rag.query("As Scoorge, what do you think of Christmas?", param=QueryParam(mode="global"))
# )

# Perform hybrid search
print(
    rag.query("As Scoorge, what do you think of Christmas?", param=QueryParam(mode="hybrid"))
)

# # Perform mix search
# print(
#     rag.query("As Scoorge, what do you think of Christmas?", param=QueryParam(mode="mix", conversation_history=empty_history))
# )