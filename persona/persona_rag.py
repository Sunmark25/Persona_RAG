import os
import argparse
import json
import networkx as nx
import copy
import asyncio
import ollama
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.operate import extract_keywords_only

def cosine_distance(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vector1: First vector
        vector2: Second vector
        
    Returns:
        float: Cosine similarity score (1.0 is most similar, 0.0 is least similar)
    """
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

def softmax(x: np.ndarray, temperature: float = 0.2) -> np.ndarray:
    """
    Apply softmax function with temperature control.
    
    Args:
        x: Input array of values
        temperature: Temperature parameter (lower = sharper distribution)
        
    Returns:
        numpy.array: Softmax probabilities
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")

    x = np.array(x)
    exp_x = np.exp(x/temperature)
    return exp_x / np.sum(exp_x)

def compute_embedding_similarity(user_input, context, embed_model = "mxbai-embed-large"):
    input_response = ollama.embed(
        model = embed_model,
        input = user_input
    )

    context_response = ollama.embed(
        model = embed_model,
        input = context
    )

    input_embedding = np.array(input_response['embeddings'][0])
    context_embedding = np.array(context_response['embeddings'][0])

    similarity_score = cosine_distance(input_embedding, context_embedding)
    return similarity_score
    
def setup_rag(working_dir):
    """Set up a LightRAG instance"""
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    return LightRAG(
        working_dir=working_dir,
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed,
    )

def load_text_chunks(chunks_file_path):
    """Load text chunks from a JSON file"""
    try:
        with open(chunks_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Cannot load text chunks from {chunks_file_path}: {e}")
        return {}

def load_graphml_to_custom_kg(graphml_path, chunks_file_path=None):
    """
    Convert a graphml file to LightRAG's custom_kg format
    
    Args:
        graphml_path: Path to the GraphML file
        chunks_file_path: (optional) Path to the text chunks JSON file
    
    Returns:
        custom_kg dictionary containing entities, relationships, and text chunks
    """
    try:
        # Read the graphml file
        G = nx.read_graphml(graphml_path)
        
        # Initialize custom_kg dictionary
        custom_kg = {
            "entities": [],
            "relationships": [],
            "chunks": []
        }
        
        # Create a set to track processed chunks
        processed_chunks = set()
        
        # If chunks_file_path is provided, load text chunks from file
        original_chunks = {}
        if chunks_file_path and os.path.exists(chunks_file_path):
            original_chunks = load_text_chunks(chunks_file_path)
            print(f"Loaded {len(original_chunks)} text chunks from {chunks_file_path}")
        
        # Process nodes (entities)
        for node_id, node_data in G.nodes(data=True):
            try:
                # Remove quotes for consistency
                entity_name = node_id.strip('"')
                entity_type = node_data.get("entity_type", "UNKNOWN").strip('"')
                description = node_data.get("description", "").strip('"')
                source_id = node_data.get("source_id", "").strip('"')
                
                entity = {
                    "entity_name": entity_name,
                    "entity_type": entity_type,
                    "description": description,
                    "source_id": source_id
                }
                custom_kg["entities"].append(entity)
                
                # Process chunks in source_id
                process_source_id_chunks(source_id, processed_chunks, custom_kg, original_chunks)
            except Exception as e:
                print(f"Error processing node {node_id}: {str(e)}")
        
        # Process edges (relationships)
        for src, tgt, edge_data in G.edges(data=True):
            try:
                # Remove quotes
                src_id = src.strip('"')
                tgt_id = tgt.strip('"')
                description = edge_data.get("description", "")
                keywords = edge_data.get("keywords", "")
                
                # Convert weight to float
                try:
                    weight = float(edge_data.get("weight", 1.0))
                except (ValueError, TypeError):
                    weight = 1.0
                
                source_id = edge_data.get("source_id", "")
                
                relationship = {
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                    "description": description,
                    "keywords": keywords,
                    "weight": weight,
                    "source_id": source_id
                }
                custom_kg["relationships"].append(relationship)
                
                # Process chunks in source_id
                process_source_id_chunks(source_id, processed_chunks, custom_kg, original_chunks)
            except Exception as e:
                print(f"Error processing relationship {src}-{tgt}: {str(e)}")
        
        # Ensure there's at least one chunk
        if not custom_kg["chunks"]:
            custom_kg["chunks"].append({
                "content": "Default chunk content for the knowledge graph",
                "source_id": "default-chunk"
            })
        
        return custom_kg
    except Exception as e:
        print(f"Error loading knowledge graph: {str(e)}")
        return None

def process_source_id_chunks(source_id, processed_chunks, custom_kg, original_chunks=None):
    """
    Process chunk IDs in the source_id field and add them to the custom_kg chunks list
    
    Args:
        source_id: Source ID string
        processed_chunks: Set of already processed chunk IDs
        custom_kg: The custom_kg dictionary to be populated
        original_chunks: (optional) Dictionary of original text chunks to get real content
    """
    if not source_id:
        return
    
    # Check if the source_id contains the <SEP> separator
    chunk_ids = source_id.split("<SEP>") if "<SEP>" in source_id else [source_id]
    
    # Process each chunk ID
    for chunk_id in chunk_ids:
        chunk_id = chunk_id.strip()
        if not chunk_id or chunk_id in processed_chunks:
            continue
        
        # Mark as processed
        processed_chunks.add(chunk_id)
        
        # Prioritize content from original text chunks
        if original_chunks and chunk_id in original_chunks:
            # Use content from original text chunks
            chunk_data = original_chunks[chunk_id].copy()  # Create a copy to avoid modifying original data
            # Ensure chunk_data has a source_id field
            if "source_id" not in chunk_data:
                chunk_data["source_id"] = chunk_id
            custom_kg["chunks"].append(chunk_data)
        else:
            # Fall back to generating content
            chunk_content = generate_chunk_content(chunk_id)
            custom_kg["chunks"].append({
                "content": chunk_content,
                "source_id": chunk_id
            })

def generate_chunk_content(chunk_id):
    """
    Generate reasonable content based on chunk ID
    Only used when original text chunks are not found
    """
    # This is a simple implementation, you can customize as needed
    if "UNKNOWN" in chunk_id:
        return "Unknown content"
    
    # If chunk ID contains semantic information, try to extract relevant content
    relevant_phrases = {
        "ANTIGONE": "Antigone is the protagonist of the play, daughter of Oedipus, who defies Creon's decree to bury her brother Polyneices.",
        "CREON": "Creon is the king of Thebes who decrees that Polyneices' body should remain unburied as punishment for his betrayal.",
        "ISMENE": "Ismene is Antigone's sister who refuses to help her bury their brother, fearing Creon's punishment.",
        "POLYNEICES": "Polyneices is Antigone's brother who fought against Thebes and whose body Creon forbids to be buried.",
        "ETEOCLES": "Eteocles is Antigone's brother who defended Thebes and was given a proper burial by Creon.",
        "HAEMON": "Haemon is Creon's son and Antigone's fiancé who pleads with his father to spare Antigone's life.",
        "TIRESIAS": "Tiresias is the blind prophet who warns Creon that the gods disapprove of his actions.",
        "CHORUS": "The Chorus represents the elders of Thebes who comment on the action and often provide moral judgments.",
        "DIVINE_LAW": "The play explores the conflict between divine law, which Antigone follows, and human law, represented by Creon's decree.",
        "BURIAL": "The burial of Polyneices is the central act of defiance in the play, representing Antigone's loyalty to family and divine law.",
        "TRAGEDY": "Antigone is a Greek tragedy by Sophocles that follows the protagonist's resistance against the state and her tragic fate.",
        "PRIDE": "Pride and stubbornness lead both Antigone and Creon to their tragic ends.",
        "LOYALTY": "The conflict between loyalty to family and loyalty to the state is a central theme of the play."
    }
    
    # Check if chunk ID contains any keywords
    for keyword, content in relevant_phrases.items():
        if keyword in chunk_id.upper():
            return content
    
    # Default content
    return f"Content related to the knowledge graph node or relationship with ID {chunk_id}"

def combine_top_k_chunks(chunks_dict, top_k=None, separator="\n\n"):
    """
    将文本块字典中的前k个文本块合并成一个字符串
    
    Args:
        chunks_dict: 包含源ID和对应文本块内容的字典 {source_id: chunk_content}
        top_k: 要合并的文本块数量，None表示全部合并
        separator: 文本块之间的分隔符
        
    Returns:
        str: 合并后的文本字符串
    """
    # 如果字典为空，返回空字符串
    if not chunks_dict:
        return ""
    
    # 获取所有文本块
    chunks_list = list(chunks_dict.items())
    
    # 如果指定了top_k且值有效，限制文本块数量
    if top_k is not None and top_k > 0:
        chunks_list = chunks_list[:min(top_k, len(chunks_list))]
    
    # 构建合并后的文本
    combined_text = ""
    for i, (source_id, content) in enumerate(chunks_list):
        if i > 0:  # 添加分隔符（除了第一个块）
            combined_text += separator
        # 可以选择添加source_id作为标识符
        combined_text += f"[{source_id}] {content}"
    
    return combined_text

def get_query_related_chunks(rag, query, param=None, top_k=None, return_combined_text=False):
    """
    获取与查询相关的所有文本块
    
    Args:
        rag: LightRAG实例 
        query: 查询文本
        param: 查询参数，默认为None时会创建一个新的QueryParam
        top_k: 要返回的文本块数量，None表示返回所有相关文本块
        return_combined_text: 是否返回合并后的文本字符串而不是字典
        
    Returns:
        dict或str: 默认返回包含源ID和对应文本块内容的字典 {source_id: chunk_content}，
                 当return_combined_text=True时返回合并后的文本字符串
    """
    # 设置默认查询参数
    if param is None:
        param = QueryParam(mode="hybrid")
    
    # 克隆param来避免修改原始对象
    debug_param = copy.deepcopy(param)
    # 启用DEBUG模式，让LightRAG返回内部信息
    debug_param.debug = True
    
    # 获取查询的关键词
    loop = asyncio.get_event_loop()
    hl_keywords, ll_keywords = loop.run_until_complete(
        extract_keywords_only(
            text=query,
            param=param,
            global_config=rag.__dict__,
            hashing_kv=rag.llm_response_cache
        )
    )
    
    print(f"查询关键词: HL={hl_keywords}, LL={ll_keywords}")
    
    # 从知识图谱中收集相关节点的source_ids
    related_source_ids = set()
    graph = rag.chunk_entity_relation_graph._graph
    
    # 根据关键词匹配节点
    for node, data in graph.nodes(data=True):
        node_str = str(node).upper()
        node_desc = str(data.get("description", "")).upper()
        
        # 检查节点名称或描述是否包含任何关键词
        if any(kw.upper() in node_str for kw in hl_keywords + ll_keywords) or \
           any(kw.upper() in node_desc for kw in hl_keywords + ll_keywords):
            
            if "source_id" in data:
                if "<SEP>" in data["source_id"]:
                    for sid in data["source_id"].split("<SEP>"):
                        related_source_ids.add(sid.strip())
                else:
                    related_source_ids.add(data["source_id"].strip())
    
    # 从边中也收集相关的source_ids
    for src, tgt, edge_data in graph.edges(data=True):
        src_str = str(src).upper()
        tgt_str = str(tgt).upper()
        desc = str(edge_data.get("description", "")).upper()
        keywords = str(edge_data.get("keywords", "")).upper()
        
        # 检查边的信息是否包含任何关键词
        if any(kw.upper() in src_str + tgt_str + desc + keywords for kw in hl_keywords + ll_keywords):
            if "source_id" in edge_data:
                if "<SEP>" in edge_data["source_id"]:
                    for sid in edge_data["source_id"].split("<SEP>"):
                        related_source_ids.add(sid.strip())
                else:
                    related_source_ids.add(edge_data["source_id"].strip())
    
    # 从text_chunks中获取文本块
    result_chunks = {}
    
    # 优先从客户端存储中获取数据(内存中的数据)
    if hasattr(rag.text_chunks, "client_storage") and "data" in rag.text_chunks.client_storage:
        chunks = rag.text_chunks.client_storage["data"]
        
        for chunk_id, chunk_data in chunks.items():
            # 直接匹配chunk_id
            if chunk_id in related_source_ids:
                result_chunks[chunk_id] = chunk_data["content"]
            # 或者匹配chunk的source_id字段
            elif "source_id" in chunk_data and chunk_data["source_id"] in related_source_ids:
                result_chunks[chunk_data["source_id"]] = chunk_data["content"]
    else:
        # 如果客户端存储不可用，从数据库中获取
        async def get_chunks():
            chunks_data = {}
            for source_id in related_source_ids:
                chunk = await rag.text_chunks.get_by_id(source_id)
                if chunk and "content" in chunk:
                    chunks_data[source_id] = chunk["content"]
            return chunks_data
        
        result_chunks = loop.run_until_complete(get_chunks())
    
    # 如果需要限制返回的文本块数量
    if top_k is not None and top_k > 0 and len(result_chunks) > top_k:
        # 转换为列表并截取前top_k个元素
        chunks_items = list(result_chunks.items())[:top_k]
        # 重建字典
        result_chunks = {k: v for k, v in chunks_items}
    
    # 返回合并后的文本字符串或原始字典
    if return_combined_text:
        return combine_top_k_chunks(result_chunks, top_k=None)  # 已经在上面限制了数量，这里不需要再次限制
    
    return result_chunks

def main():
    # Define all five personalities
    personalities = ["agreeableness", "conscientiousness", "extraversion", "neuroticism", "openness"]
    
    # Dictionary to store LightRAG instances for each personality
    rag_instances = {}
    
    # Set up each personality's model
    for personality in personalities:
        # Create directory in big_five if it doesn't exist
        big_five_dir = f"./big_five/{personality}"
        if not os.path.exists(big_five_dir):
            os.makedirs(big_five_dir)
        
        # Define paths for source and destination files
        source_graphml = f"./antigone/{personality}/graph_chunk_entity_relation.graphml"
        source_chunks = f"./antigone/{personality}/kv_store_text_chunks.json"
        
        # dest_graphml = f"./persona/big_five/{personality}/graph_chunk_entity_relation.graphml"
        # dest_chunks = f"./persona/big_five/{personality}/kv_store_text_chunks.json"
        
        # Set up working directory for this personality
        working_dir = f"./big_five/{personality}"
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)
        
        # Set up LightRAG instance for this personality
        rag = setup_rag(working_dir)
        
        # Load knowledge graph
        custom_kg = load_graphml_to_custom_kg(source_graphml, source_chunks)
        if custom_kg:
            rag.insert_custom_kg(custom_kg)
            print(f"[{personality}] Loaded {len(custom_kg['entities'])} entities, {len(custom_kg['relationships'])} relationships, and {len(custom_kg['chunks'])} text chunks")
            rag_instances[personality] = rag
        else:
            print(f"WARNING: Failed to load knowledge graph for {personality}")
    
    # Fixed empty conversation history
    empty_history = []
    
    # Define Antigone questions
    antigone_questions = [
        "As Antigone, what do you believe happens to an unburied soul in the afterlife?",
        "If you could speak to your father Oedipus now, what would you ask him about defying authority?",
        "What do you tell yourself when faced with the choice between family loyalty and obedience to the state?",
        "In what ways has your family's tragic history prepared you for making sacrifices others would not make?",
        "What frightens you more: an unmarked grave or a life of compromise?",
        "How do you reconcile your devotion to divine law with the suffering your actions bring to those who love you?",
        "If you knew with certainty that Polyneices would not have done the same for you, would your decision change?",
        "What parts of yourself do you recognize in Creon that you refuse to acknowledge?",
        "How has being a daughter of Oedipus shaped your willingness to stand alone against consensus?",
        "When you imagine your name echoing through future generations, what do you hope they will say of your choice?",
        "What gives you the certainty that the gods approve of your actions when even the priests remain silent?",
        "If Ismene had joined you, would your sacrifice feel more or less meaningful?"
    ]
    
    # System prompt for each personality
    system_prompt = "Please answer the following question as Antigone with your personality traits. "
    
    # Execute queries for each question
    print("\n=== Generating Answers for All Personalities ===")
    for question in antigone_questions:
        print(f"\nQuestion: {question}")
        print("-" * 80)
        response_dict = {}
        
        # Generate answers for each personality and calculate similarity
        for personality in personalities:
            rag = rag_instances.get(personality)
            if not rag:
                print(f"Skipping {personality} as its RAG model couldn't be loaded")
                continue
                
            print(f"\n--- {personality.capitalize()} Personality ---")
            
            # Generate answer
            result = rag.query(
                system_prompt + question, 
                param=QueryParam(mode="hybrid", conversation_history=empty_history)
            )
            print(f"Answer: {result}")
            
            # Get related text chunks and calculate similarity
            combined_text = get_query_related_chunks(rag, question, top_k=20, return_combined_text=True)
            similarity_score = compute_embedding_similarity(question, combined_text)
            print(f"Question-Context Similarity: {similarity_score:.4f}")

            response_dict.update({"personality": personality, "response":result, "similarity": similarity_score})
            
            # Print the first 200 characters of combined text for reference
            print(f"Context (excerpt): {combined_text[:200]}..." if len(combined_text) > 200 else combined_text)
        
        print("=" * 80)

# If this script is run directly, execute the main function
if __name__ == "__main__":
    main()
