import os
import argparse
import json
import networkx as nx

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed


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
        "SCROOGE": "Ebenezer Scrooge was a miserly businessman who valued money over human connection.",
        "CHRISTMAS": "Christmas is a time for joy, generosity, and family gatherings.",
        "TINY_TIM": "Tiny Tim was a small, ill boy who remained cheerful despite his difficulties.",
        "MARLEY": "Jacob Marley was Scrooge's deceased business partner who returned as a ghost.",
        "GHOST": "The spirits of Christmas visited Scrooge to help him change his ways.",
        "CRATCHIT": "Bob Cratchit was Scrooge's underpaid clerk who maintained a cheerful disposition.",
        "TRANSFORMATION": "Scrooge underwent a profound transformation from miserly to generous.",
        "REDEMPTION": "The story follows Scrooge's journey of redemption and change of heart."
    }
    
    # Check if chunk ID contains any keywords
    for keyword, content in relevant_phrases.items():
        if keyword in chunk_id.upper():
            return content
    
    # Default content
    return f"Content related to the knowledge graph node or relationship with ID {chunk_id}"

def main():
    working_dir = "./neuroticism"
    graphml_path = "./big_five/neuroticism/graph_chunk_entity_relation.graphml"
    chunks_path = "./big_five/neuroticism/kv_store_text_chunks.json"
    
    # Set up LightRAG instance
    rag = setup_rag(working_dir)
    
    # Load knowledge graph
    custom_kg = load_graphml_to_custom_kg(graphml_path, chunks_path)
    if custom_kg:
        rag.insert_custom_kg(custom_kg)
        print(f"Loaded {len(custom_kg['entities'])} entities, {len(custom_kg['relationships'])} relationships, and {len(custom_kg['chunks'])} text chunks")
    
    # Fixed empty conversation history
    empty_history = []
    
    scrooge_questions = [
        "As Scoorge, what do you think of Christmas?",
        "If you could speak to your younger self, what would you warn him about trusting others?",
        "What do you tell yourself when you see someone in need that allows you to walk past them?",
        "In what ways has accumulating wealth protected you from feeling vulnerable?",
        "What frightens you more: losing your fortune or losing your isolation?",
        "How do you reconcile your business practices with your understanding of morality?",
        "If you knew with certainty that no one would ever take advantage of your generosity, would your behavior change?",
    ]
    
    # # Execute global search
    # print("\n=== GLOBAL Mode Query Results ===")
    # for question in scrooge_questions:
    #     print(f"\nQuestion: {question}")
    #     result = rag.query(question, param=QueryParam(mode="global", conversation_history=empty_history))
    #     print(f"Answer: {result}")
    #     print("-" * 80)

    system_prompt = "Please answer the following questions as character Scoorge. "
    
    # Execute hybrid search
    print("\n=== HYBRID Mode Query Results ===")
    for question in scrooge_questions:
        print(f"\nQuestion: {question}")
        result = rag.query(system_prompt + question, param=QueryParam(mode="hybrid", conversation_history=empty_history))
        print(f"Answer: {result}")
        print("-" * 80)

# If this script is run directly, execute the main function
if __name__ == "__main__":
    main()