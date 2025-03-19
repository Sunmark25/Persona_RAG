# Entire Project Workflow

First, obtain the file you need to extract (must be in txt format, like a novel or play).
Use the following code to create a LightRAG object and generate a general graphml file `original_rag.py`:

```python
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
```

The directory and file can be replaced according to your needs.

Next, you need to modify the character name that needs to be parsed in `lightrag/prompt.py`:

```python
PROMPTS["TARGET_CHARACTER"] = "Scoorge"  # Hardcoded character name
```

Then you need to create 5 directories under `WORKING_DIR`, namely:
+ agreeableness
+ conscientiousness
+ extraversion
+ neuroticism
+ openness

Next, you need to execute 5 commands to decompose the general graph into each Personality Trait Subgraph Directory:

```bash
python extract_subgraph.py persona/graph_chunk_entity_relation.graphml big_five/neuroticism '"HIGH_OPENNESS"' '"LOW_OPENNESS"' --source_chunks_file persona/kv_store_text_chunks.json
```

Basic usage with minimum required parameters:

```bash
python3 script_name.py knowledge_graph.graphml output_dir node1_id
```

Advanced usage with all optional parameters:

```bash
python3 script_name.py knowledge_graph.graphml output_dir node1_id node2_id --distance 3 --source_text source.txt --source_chunks_file original_chunks.json
```

Then when using `persona_rag.py`, create 5 different directories to generate 5 different embeddings for the different personality traits.