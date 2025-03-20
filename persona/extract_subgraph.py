#!/usr/bin/env python3

import networkx as nx
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from lightrag.utils import compute_mdhash_id
from lightrag.operate import chunking_by_token_size

def create_doc_status(doc_id, content_summary, content_length):
    """Create document status entry"""
    return {
        "status": "PROCESSED",
        "content_summary": content_summary,
        "content_length": content_length,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }

def create_chunks(text, doc_id):
    """Create text chunks from source text"""
    chunks = {}
    for chunk in chunking_by_token_size(
        text,
        overlap_token_size=100,
        max_token_size=1200,
        tiktoken_model="gpt-4"
    ):
        chunk_id = compute_mdhash_id(chunk["content"], prefix="chunk-")
        chunks[chunk_id] = {
            "content": chunk["content"],
            "full_doc_id": doc_id
        }
    return chunks

def create_vector_data(subgraph):
    """Create vector data for entities and relationships"""
    entities = {}
    relationships = {}
    
    # Process nodes/entities
    for node in subgraph.nodes(data=True):
        entity_id = compute_mdhash_id(node[0], prefix="ent-")
        entities[entity_id] = {
            "content": node[0] + (node[1].get("description", "") or ""),
            "entity_name": node[0]
        }
    
    # Process edges/relationships
    for edge in subgraph.edges(data=True):
        rel_id = compute_mdhash_id(edge[0] + edge[1], prefix="rel-")
        # 获取或创建source_id
        source_id = edge[2].get("source_id", "")
        if not source_id:
            # 如果没有source_id，创建一个默认的基于关系的source_id
            source_id = f"default-rel-{edge[0]}-{edge[1]}"
            print(f"为关系 {edge[0]}-{edge[1]} 创建默认source_id: {source_id}")
        
        relationships[rel_id] = {
            "src_id": edge[0],
            "tgt_id": edge[1],
            "content": (edge[2].get("description", "") or "") + 
                      (edge[2].get("keywords", "") or ""),
            "source_id": source_id  # 确保source_id被包含在关系数据中
        }
    
    return entities, relationships

def collect_source_ids(graph):
    """
    Collect all source_ids from nodes and edges in the graph.
    
    Args:
        graph: NetworkX graph
        
    Returns:
        Set of all source_ids found in the graph
    """
    source_ids = set()
    
    # Collect from nodes
    for _, node_data in graph.nodes(data=True):
        if 'source_id' in node_data and node_data['source_id']:
            # Handle SEP-separated IDs
            if '<SEP>' in node_data['source_id']:
                for chunk_id in node_data['source_id'].split('<SEP>'):
                    if chunk_id.strip():
                        source_ids.add(chunk_id.strip())
            else:
                source_ids.add(node_data['source_id'].strip())
    
    # Collect from edges
    for _, _, edge_data in graph.edges(data=True):
        if 'source_id' in edge_data and edge_data['source_id']:
            # Handle SEP-separated IDs
            if '<SEP>' in edge_data['source_id']:
                for chunk_id in edge_data['source_id'].split('<SEP>'):
                    if chunk_id.strip():
                        source_ids.add(chunk_id.strip())
            else:
                source_ids.add(edge_data['source_id'].strip())
    
    return source_ids

def load_text_chunks(chunks_file_path):
    """
    Load text chunks from a file.
    
    Args:
        chunks_file_path: Path to the text chunks JSON file
        
    Returns:
        Dictionary of chunk_id -> chunk_data
    """
    try:
        with open(chunks_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load text chunks from {chunks_file_path}: {e}")
        return {}

def extract_relevant_chunks(chunks_dict, source_ids):
    """
    Extract chunks related to the given source IDs.
    
    Args:
        chunks_dict: Dictionary of all chunks
        source_ids: Set of source IDs to extract
        
    Returns:
        Dictionary of chunk_id -> chunk_data for relevant chunks
    """
    relevant_chunks = {}
    
    # Direct match by chunk ID
    for chunk_id, chunk_data in chunks_dict.items():
        if chunk_id in source_ids:
            relevant_chunks[chunk_id] = chunk_data
    
    # Also match chunks where source_id is in the chunk data
    for chunk_id, chunk_data in chunks_dict.items():
        if 'source_id' in chunk_data and chunk_data['source_id'] in source_ids:
            relevant_chunks[chunk_id] = chunk_data
    
    return relevant_chunks

def extract_subgraph(graph, nodes, stopping_node=None, max_distance=2):
    """
    Extract a subgraph containing nodes within max_distance of given nodes.
    If stopping_node is provided, it will be included but its neighbors won't be explored.
    
    Args:
        graph: NetworkX graph
        nodes: List of node IDs to use as focal points
        stopping_node: Node ID that should be included but whose neighbors should not be explored
        max_distance: Maximum distance from focal nodes (default: 2)
    
    Returns:
        NetworkX graph containing the subgraph
    """
    # 验证节点是否存在
    for node in nodes:
        if node not in graph:
            raise ValueError(f"Node {node} not found in graph")
    if stopping_node and stopping_node not in graph:
        raise ValueError(f"Stopping node {stopping_node} not found in graph")
    
    # 用于存储最终子图节点的集合
    subgraph_nodes = set(nodes)
    
    # 广度优先搜索实现
    # 我们将跟踪与焦点节点的距离
    current_nodes = set(nodes)  # 从距离为0的焦点节点开始
    visited = set(nodes)
    
    # 探索直到最大距离
    for distance in range(1, max_distance + 1):
        next_nodes = set()
        
        for node in current_nodes:
            # 如果是停止节点，跳过探索其邻居
            if node == stopping_node:
                continue
            
            # 添加未访问的邻居到下一层
            neighbors = set(graph.neighbors(node)) - visited
            next_nodes.update(neighbors)
            visited.update(neighbors)
        
        # 更新当前节点集合，用于下一次迭代
        current_nodes = next_nodes
        
        # 更新子图节点
        subgraph_nodes.update(next_nodes)
        
        # 如果没有更多节点需要探索，则退出循环
        if not current_nodes:
            break
    
    # 创建子图
    return graph.subgraph(subgraph_nodes).copy()

def main():
    # 设置参数解析器
    parser = argparse.ArgumentParser(
        description='Extract subgraph and generate LightRAG files'
    )
    parser.add_argument('input_file', type=str, help='Input graphml file path')
    parser.add_argument('working_dir', type=str, help='Output directory for all LightRAG files')
    parser.add_argument('nodes', type=str, nargs='+', help='1-2 node IDs to use as focal points')
    parser.add_argument('--distance', type=int, default=2,
                      help='Maximum distance from focal nodes (default: 2)')
    parser.add_argument('--stopping_node', type=str, 
                      help='Node to include but not explore its neighbors')
    parser.add_argument('--source_text', type=str, help='Optional source text file to include')
    parser.add_argument('--source_chunks_file', type=str, 
                      help='Optional file path to original text chunks JSON to extract from')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate number of nodes
    if len(args.nodes) > 2:
        raise ValueError("Maximum of 2 focal nodes allowed")
    
    # Create working directory
    working_dir = Path(args.working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and process graph
    try:
        # Load input graph
        graph = nx.read_graphml(args.input_file)
        print(f"Loaded original graph: {len(graph)} nodes, {len(graph.edges)} edges")
        
        # 提取子图，传入停止节点参数
        subgraph = extract_subgraph(graph, args.nodes, args.stopping_node, args.distance)
        print(f"Extracted subgraph: {len(subgraph)} nodes, {len(subgraph.edges)} edges")
        
        # Save graphml format
        graphml_path = working_dir / "graph_chunk_entity_relation.graphml"
        nx.write_graphml(subgraph, graphml_path)
        print(f"Saved GraphML to {graphml_path}")
        
        # Save JSON graph data
        graph_data_path = working_dir / "graph_data.json"
        with open(graph_data_path, "w") as f:
            json.dump(nx.node_link_data(subgraph), f, indent=2)
        print(f"Saved graph data to {graph_data_path}")
        
        # Generate vector data
        entities, relationships = create_vector_data(subgraph)
        
        # Save vector data
        entities_path = working_dir / "vdb_entities.json"
        with open(entities_path, "w") as f:
            json.dump(entities, f, indent=2)
        print(f"Saved entity vectors to {entities_path}")
        
        relationships_path = working_dir / "vdb_relationships.json"
        with open(relationships_path, "w") as f:
            json.dump(relationships, f, indent=2)
        print(f"Saved relationship vectors to {relationships_path}")
        
        # Process source text if provided
        if args.source_text:
            try:
                with open(args.source_text) as f:
                    text = f.read()
                doc_id = compute_mdhash_id(text, prefix="doc-")
                
                # Create and save document status
                doc_status = {
                    doc_id: create_doc_status(
                        doc_id=doc_id,
                        content_summary=text[:100] + "..." if len(text) > 100 else text,
                        content_length=len(text)
                    )
                }
                status_path = working_dir / "kv_store_doc_status.json"
                with open(status_path, "w") as f:
                    json.dump(doc_status, f, indent=2)
                print(f"Saved document status to {status_path}")
                
                # Save full document
                full_docs_path = working_dir / "kv_store_full_docs.json"
                with open(full_docs_path, "w") as f:
                    json.dump({doc_id: {"content": text}}, f, indent=2)
                print(f"Saved full document to {full_docs_path}")
                
                # Create and save text chunks
                chunks = create_chunks(text, doc_id)
                chunks_path = working_dir / "kv_store_text_chunks.json"
                with open(chunks_path, "w") as f:
                    json.dump(chunks, f, indent=2)
                print(f"Saved text chunks to {chunks_path}")
                
                # Save chunks vector data
                chunks_vdb_path = working_dir / "vdb_chunks.json"
                with open(chunks_vdb_path, "w") as f:
                    json.dump(chunks, f, indent=2)
                print(f"Saved chunks vector data to {chunks_vdb_path}")
            
            except Exception as e:
                print(f"Warning: Error processing source text: {e}")
        
        # Extract and save text chunks if source_chunks_file is provided
        if args.source_chunks_file:
            try:
                # Collect source IDs from the subgraph
                source_ids = collect_source_ids(subgraph)
                print(f"Collected {len(source_ids)} unique source IDs from the subgraph")
                
                # Load original text chunks
                original_chunks = load_text_chunks(args.source_chunks_file)
                print(f"Loaded {len(original_chunks)} text chunks from {args.source_chunks_file}")
                
                # Extract relevant chunks
                relevant_chunks = extract_relevant_chunks(original_chunks, source_ids)
                print(f"Extracted {len(relevant_chunks)} relevant text chunks")
                
                # Save relevant chunks
                if relevant_chunks:
                    chunks_path = working_dir / "kv_store_text_chunks.json"
                    with open(chunks_path, "w") as f:
                        json.dump(relevant_chunks, f, indent=2)
                    print(f"Saved {len(relevant_chunks)} text chunks to {chunks_path}")
                    
                    # Also save as vdb_chunks.json
                    chunks_vdb_path = working_dir / "vdb_chunks.json"
                    with open(chunks_vdb_path, "w") as f:
                        json.dump(relevant_chunks, f, indent=2)
                    print(f"Saved chunks vector data to {chunks_vdb_path}")
                else:
                    print("Warning: No relevant text chunks were found in the original file.")
            except Exception as e:
                print(f"Warning: Error extracting text chunks: {e}")
        
        # Ensure all source_ids have corresponding text chunks
        # This adds default text chunks for any relationships with source_ids that don't have corresponding text chunks
        all_source_ids = set()
        default_chunks = {}
        
        # Collect source_ids from all relationships
        for rel_id, rel_data in relationships.items():
            if "source_id" in rel_data and rel_data["source_id"]:
                source_ids = rel_data["source_id"].split("<SEP>") if "<SEP>" in rel_data["source_id"] else [rel_data["source_id"]]
                for source_id in source_ids:
                    source_id = source_id.strip()
                    if source_id:
                        all_source_ids.add(source_id)
        
        # Load existing chunks 
        # First try from the working directory if it exists
        existing_chunks = {}
        chunks_path = working_dir / "kv_store_text_chunks.json"
        if chunks_path.exists():
            try:
                with open(chunks_path, "r") as f:
                    existing_chunks = json.load(f)
                print(f"Loaded {len(existing_chunks)} existing text chunks from {chunks_path}")
            except Exception as e:
                print(f"Warning: Failed to load text chunks from {chunks_path}: {e}")
        
        # Create default text chunks for source_ids that don't have chunks
        count_default_chunks = 0
        for source_id in all_source_ids:
            if source_id not in existing_chunks:
                # Find which relationships use this source_id
                rel_info = []
                for rel_id, rel_data in relationships.items():
                    if rel_data.get("source_id") == source_id or (
                        "<SEP>" in rel_data.get("source_id", "") and 
                        source_id in rel_data.get("source_id", "").split("<SEP>")
                    ):
                        rel_info.append(f"{rel_data.get('src_id')} -> {rel_data.get('tgt_id')}: {rel_data.get('content', '')}")
                
                # Create a default chunk with relationship information
                rel_text = "\n".join(rel_info) if rel_info else f"Default content for source_id: {source_id}"
                default_chunks[source_id] = {
                    "content": f"默认文本块 - 用于关系: {rel_text}",
                    "source_id": source_id
                }
                count_default_chunks += 1
        
        # If we created any default chunks, update the chunks files
        if default_chunks:
            # Update existing chunks with default chunks
            existing_chunks.update(default_chunks)
            print(f"Created {count_default_chunks} default text chunks for relationships without text blocks")
            
            # Save the updated chunks
            with open(chunks_path, "w") as f:
                json.dump(existing_chunks, f, indent=2)
            print(f"Updated text chunks at {chunks_path}")
            
            # Also update vdb_chunks.json if it exists
            chunks_vdb_path = working_dir / "vdb_chunks.json"
            if chunks_vdb_path.exists():
                try:
                    with open(chunks_vdb_path, "r") as f:
                        vdb_chunks = json.load(f)
                    vdb_chunks.update(default_chunks)
                    with open(chunks_vdb_path, "w") as f:
                        json.dump(vdb_chunks, f, indent=2)
                    print(f"Updated chunks vector data at {chunks_vdb_path}")
                except Exception as e:
                    print(f"Warning: Failed to update {chunks_vdb_path}: {e}")
            else:
                # Create vdb_chunks.json if it doesn't exist
                with open(chunks_vdb_path, "w") as f:
                    json.dump(existing_chunks, f, indent=2)
                print(f"Created chunks vector data at {chunks_vdb_path}")
        
        # Create empty LLM cache
        cache_path = working_dir / "kv_store_llm_response_cache.json"
        with open(cache_path, "w") as f:
            json.dump({}, f, indent=2)
        print(f"Created empty LLM cache at {cache_path}")
        
    except Exception as e:
        raise Exception(f"Error during processing: {e}")

if __name__ == "__main__":
    main()
