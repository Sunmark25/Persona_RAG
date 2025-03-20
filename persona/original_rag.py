import os

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

WORKING_DIR = "./antigone"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete,
    embedding_func=openai_embed,
    # llm_model_func=gpt_4o_complete
)


with open("./antigone/antigone.txt", "r", encoding="utf-8") as f:
    rag.insert(f.read())