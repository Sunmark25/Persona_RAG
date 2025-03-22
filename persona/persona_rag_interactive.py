import os
import gradio as gr
import asyncio
import json
import numpy as np
import ollama
import copy
import networkx as nx
from typing import Dict, List, Optional, Tuple, Union
import time

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.operate import extract_keywords_only

# 全局变量用于存储预加载的RAG实例
GLOBAL_RAG_INSTANCES = {}

# 保留原有prompt以用于集成回答
integration_prompt = """
# Personality Response Integration System

You are a sophisticated character response integration system. You will be provided with:

1. A question or scenario
2. Five different responses to this question, each reflecting a different Big Five personality dimension (Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism)
3. A weight assigned to each response (with all weights summing to 1.0)

Your task is to create a single, coherent, and natural response that integrates these five perspectives according to their assigned weights. This is not simply copying and pasting sections from each response, but thoughtfully blending the content, tone, concerns, and perspectives from each dimension into a unified character voice.

## Integration Guidelines

When blending the responses:

1. Higher weighted traits should have more influence on:
   - The overall tone and style of the response
   - The main points and recommendations made
   - The decision-making approach and priorities expressed
   - The emotional coloring of the response

2. Pay special attention to what each personality dimension contributes:
   - **Openness**: Creative ideas, curiosity, appreciation for novelty, abstract thinking
   - **Conscientiousness**: Organization, thoroughness, responsibility, planning, attention to detail
   - **Extraversion**: Social energy, enthusiasm, assertiveness, positive emotions
   - **Agreeableness**: Empathy, cooperation, consideration of others, conflict avoidance
   - **Neuroticism**: Awareness of risks, emotional sensitivity, caution, concern for problems

3. Create natural transitions between different personality aspects, avoiding abrupt shifts in tone or perspective

4. The final response should read as if written by a single coherent character, not as disconnected perspectives

## Output Format

Integrated Character Response:
[Your integrated response that blends all five perspectives according to their weights while maintaining a coherent character voice]
"""

# 保留原有函数
def cosine_distance(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

def softmax(x: np.ndarray, temperature: float = 0.2) -> np.ndarray:
    """Apply softmax function with temperature control."""
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    x = np.array(x)
    exp_x = np.exp(x/temperature)
    return exp_x / np.sum(exp_x)

def compute_embedding_similarity(user_input, context, embed_model = "mxbai-embed-large"):
    """Compute embedding similarity between two text pieces."""
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

# 从原始文件复制所需函数
def load_text_chunks(chunks_file_path):
    """Load text chunks from a JSON file"""
    try:
        with open(chunks_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Cannot load text chunks from {chunks_file_path}: {e}")
        return {}

def process_source_id_chunks(source_id, processed_chunks, custom_kg, original_chunks=None):
    """
    Process chunk IDs in the source_id field and add them to the custom_kg chunks list
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

def load_graphml_to_custom_kg(graphml_path, chunks_file_path=None):
    """
    Convert a graphml file to LightRAG's custom_kg format
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

def get_query_related_chunks(rag, query, param=None, top_k=None, return_combined_text=False):
    """
    获取与查询相关的所有文本块
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

# 初始化RAG实例的函数
def initialize_rag_instances():
    """初始化所有人格的RAG实例，保存在全局变量中以供复用"""
    global GLOBAL_RAG_INSTANCES
    
    personalities = ["agreeableness", "conscientiousness", "extraversion", "neuroticism", "openness"]
    
    print("Initializing RAG instances for all personalities...")
    for personality in personalities:
        # 创建big_five目录（如果不存在）
        big_five_dir = f"./big_five/{personality}"
        if not os.path.exists(big_five_dir):
            os.makedirs(big_five_dir)
        
        # 设置源文件路径
        source_graphml = f"./antigone/{personality}/graph_chunk_entity_relation.graphml"
        source_chunks = f"./antigone/{personality}/kv_store_text_chunks.json"
        
        # 设置工作目录
        working_dir = f"./big_five/{personality}"
        
        # 设置RAG实例
        rag = setup_rag(working_dir)
        
        # 加载知识图谱
        custom_kg = load_graphml_to_custom_kg(source_graphml, source_chunks)
        if custom_kg:
            rag.insert_custom_kg(custom_kg)
            print(f"[{personality}] Loaded {len(custom_kg['entities'])} entities, {len(custom_kg['relationships'])} relationships, and {len(custom_kg['chunks'])} text chunks")
            GLOBAL_RAG_INSTANCES[personality] = rag
        else:
            print(f"WARNING: Failed to load knowledge graph for {personality}")
    
    print("All RAG instances initialized!")

def generate_response(user_input, temperature=0.2):
    """生成所有人格的回答及集成回答"""
    global GLOBAL_RAG_INSTANCES
    
    # 检查RAG实例是否已初始化
    if not GLOBAL_RAG_INSTANCES:
        print("Initializing RAG instances...")
        initialize_rag_instances()
    
    # 定义人格列表和空的回答字典
    personalities = ["agreeableness", "conscientiousness", "extraversion", "neuroticism", "openness"]
    response_dict = {}
    similarity_scores = np.array([])
    
    # 系统提示
    system_prompt = "Please answer the following question as Antigone with your personality traits. "
    
    # 空对话历史
    empty_history = []
    
    print(f"Generating responses for query: {user_input}")
    
    # 为每个人格生成回答
    for personality in personalities:
        rag = GLOBAL_RAG_INSTANCES.get(personality)
        if not rag:
            print(f"Skipping {personality} as its RAG model couldn't be loaded")
            continue
            
        print(f"Generating response for {personality}...")
        
        # 生成回答
        start_time = time.time()
        result = rag.query(
            system_prompt + user_input, 
            param=QueryParam(mode="hybrid", conversation_history=empty_history)
        )
        
        # 获取相关文本块并计算相似度
        combined_text = get_query_related_chunks(rag, user_input, top_k=20, return_combined_text=True)
        similarity_score = compute_embedding_similarity(result, combined_text)
        print(f"{personality} response generated in {time.time() - start_time:.2f} seconds. Similarity: {similarity_score:.4f}")
        
        # 存储结果
        response_dict[personality] = (result, similarity_score)
        similarity_scores = np.append(similarity_scores, similarity_score)
    
    # 应用softmax得到概率
    probabilities = softmax(similarity_scores, temperature=temperature)
    print(f"Softmax probabilities: {probabilities}")
    
    # 创建personality到softmax概率的映射
    personality_to_prob = {}
    for i, personality in enumerate([p for p in personalities if p in response_dict]):
        personality_to_prob[personality] = probabilities[i]
    
    # 准备构建集成提示的数据
    prompt_data = {
        "o_weight": personality_to_prob.get("openness", 0.0),
        "c_weight": personality_to_prob.get("conscientiousness", 0.0),
        "e_weight": personality_to_prob.get("extraversion", 0.0),
        "a_weight": personality_to_prob.get("agreeableness", 0.0),
        "n_weight": personality_to_prob.get("neuroticism", 0.0)
    }
    
    # 构建集成提示
    integration_input = f"""
Question: {user_input}

Openness Response (Weight: {prompt_data['o_weight']:.4f}):
{response_dict.get('openness', ('N/A', 0.0))[0]}

Conscientiousness Response (Weight: {prompt_data['c_weight']:.4f}):
{response_dict.get('conscientiousness', ('N/A', 0.0))[0]}

Extraversion Response (Weight: {prompt_data['e_weight']:.4f}):
{response_dict.get('extraversion', ('N/A', 0.0))[0]}

Agreeableness Response (Weight: {prompt_data['a_weight']:.4f}):
{response_dict.get('agreeableness', ('N/A', 0.0))[0]}

Neuroticism Response (Weight: {prompt_data['n_weight']:.4f}):
{response_dict.get('neuroticism', ('N/A', 0.0))[0]}
"""
    
    # 调用GPT-4o-mini生成集成回答
    print("Generating integrated response...")
    integrated_response = gpt_4o_mini_complete(
        system_prompt=integration_prompt,
        user_prompt=integration_input
    )
    
    # 组织结果返回
    results = {
        "integrated_response": integrated_response,
        "personality_responses": {},
        "weights": {}
    }
    
    for personality in personalities:
        if personality in response_dict:
            results["personality_responses"][personality] = response_dict[personality][0]
            results["weights"][personality] = personality_to_prob.get(personality, 0.0)
    
    return results

# Gradio界面相关函数
def chat_with_antigone(message, history, temperature):
    """处理用户输入并返回回答"""
    if not message:
        return "", history, "", "", "", "", "", "", "", "", "", ""
    
    # 生成回答
    try:
        results = generate_response(message, temperature=float(temperature))
        
        integrated_response = results["integrated_response"]
        
        # 获取各个人格的回答
        o_response = results["personality_responses"].get("openness", "N/A")
        o_weight = results["weights"].get("openness", 0.0)
        
        c_response = results["personality_responses"].get("conscientiousness", "N/A")
        c_weight = results["weights"].get("conscientiousness", 0.0)
        
        e_response = results["personality_responses"].get("extraversion", "N/A")
        e_weight = results["weights"].get("extraversion", 0.0)
        
        a_response = results["personality_responses"].get("agreeableness", "N/A")
        a_weight = results["weights"].get("agreeableness", 0.0)
        
        n_response = results["personality_responses"].get("neuroticism", "N/A")
        n_weight = results["weights"].get("neuroticism", 0.0)
        
        # 更新历史记录
        history = history + [(message, integrated_response)]
        
        return "", history, o_response, f"{o_weight:.4f}", c_response, f"{c_weight:.4f}", e_response, f"{e_weight:.4f}", a_response, f"{a_weight:.4f}", n_response, f"{n_weight:.4f}"
    
    except Exception as e:
        error_message = f"Error generating response: {str(e)}"
        print(error_message)
        return "", history + [(message, error_message)], "", "", "", "", "", "", "", "", "", ""

# 创建Gradio界面
with gr.Blocks(title="Antigone Personality-Weighted Responses") as demo:
    gr.Markdown("# Antigone: Personality-Weighted RAG Integration")
    gr.Markdown("Interact with Antigone, whose responses blend five personality traits based on content similarity.")
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=600, label="Conversation with Antigone", type="messages")
            msg = gr.Textbox(
                placeholder="Ask Antigone a question...",
                label="Your Message",
                lines=2
            )
            with gr.Row():
                temperature = gr.Slider(
                    minimum=0.01, maximum=1.0, value=0.2, step=0.01,
                    label="Temperature (lower values = sharper personality contrasts)"
                )
                clear = gr.Button("Clear Conversation")
        
        with gr.Column(scale=2):
            with gr.Tab("Openness (Creative)"):
                openness_response = gr.Textbox(label="Response", lines=12)
                openness_weight = gr.Textbox(label="Weight")
                
            with gr.Tab("Conscientiousness (Organized)"):
                consc_response = gr.Textbox(label="Response", lines=12)
                consc_weight = gr.Textbox(label="Weight")
                
            with gr.Tab("Extraversion (Outgoing)"):
                extra_response = gr.Textbox(label="Response", lines=12)
                extra_weight = gr.Textbox(label="Weight")
                
            with gr.Tab("Agreeableness (Empathetic)"):
                agree_response = gr.Textbox(label="Response", lines=12)
                agree_weight = gr.Textbox(label="Weight")
                
            with gr.Tab("Neuroticism (Cautious)"):
                neuro_response = gr.Textbox(label="Response", lines=12)
                neuro_weight = gr.Textbox(label="Weight")
    
    # 事件绑定
    msg.submit(
        chat_with_antigone,
        [msg, chatbot, temperature],
        [msg, chatbot, openness_response, openness_weight, consc_response, consc_weight, 
         extra_response, extra_weight, agree_response, agree_weight, neuro_response, neuro_weight]
    )
    
    clear.click(lambda: ([], "", "", "", "", "", "", "", "", "", "", ""), 
                None, 
                [chatbot, msg, openness_response, openness_weight, consc_response, consc_weight, 
                 extra_response, extra_weight, agree_response, agree_weight, neuro_response, neuro_weight])
    
    gr.Markdown("""
    ### How It Works
    
    This demo uses RAG (Retrieval-Augmented Generation) with different personality biases:
    
    1. Your question retrieves relevant context from Antigone's knowledge base
    2. Five different personality models generate distinct responses
    3. Each response's similarity to the retrieved context is measured
    4. Softmax converts similarities to probability weights
    5. The final response integrates all personalities according to these weights
    
    **Adjust the temperature slider** to control how much the weights favor the highest similarity.
    Lower values create sharper contrasts between personalities.
    """)

# 启动程序
if __name__ == "__main__":
    # 初始化rag实例
    initialize_rag_instances()
    
    # 启动Gradio界面
    demo.launch(share=True)
