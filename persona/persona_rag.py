import os
import argparse
import json
import networkx as nx
import copy
import asyncio
import ollama
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from openai import OpenAI


from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.operate import extract_keywords_only

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

## Input Format

You will receive input in this format:

Question: [The original question or scenario]

Openness Response (Weight: {o_weight}):
[Full response from Openness perspective]

Conscientiousness Response (Weight: {c_weight}):
[Full response from Conscientiousness perspective]

Extraversion Response (Weight: {e_weight}):
[Full response from Extraversion perspective]

Agreeableness Response (Weight: {a_weight}):
[Full response from Agreeableness perspective]

Neuroticism Response (Weight: {n_weight}):
[Full response from Neuroticism perspective]

## Output Format

Provide your response in this format:

Integrated Character Response:
[Your integrated response that blends all five perspectives according to their weights while maintaining a coherent character voice]

Integration Process:
[A brief explanation of how you blended the responses, noting which traits had the strongest influence and how they shaped the final response]

## Integration Examples

Example of high-weight integration:
- When a trait has a weight of 0.4 or higher, its perspective should dominate the response
- If Openness has a weight of 0.5, the final response should primarily reflect creative, curious, and exploratory thinking, while still incorporating smaller elements from other traits

Example of balanced integration:
- When weights are more evenly distributed (e.g., 0.25, 0.25, 0.2, 0.15, 0.15), create a balanced response that harmonizes the different perspectives
- Ensure the traits with 0.25 weights have slightly more influence than those with 0.15 weights

Example of minimal-weight integration:
- Traits with very low weights (0.1 or less) should still have a subtle presence in the response
- If Neuroticism has a weight of 0.05, perhaps just a brief mention of a possible concern or a slight note of caution in an otherwise positive response

Remember that this is an exercise in creating a realistic, multidimensional character response that reflects a specific personality profile through the weighted integration of different trait perspectives.
"""

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
    # If input or context is empty, return default similarity value
    if not user_input or not context:
        print("Warning: Input or context is empty, cannot compute similarity")
        return 0.5  # Return a moderate default value
    
    try:
        input_response = ollama.embed(
            model = embed_model,
            input = user_input
        )

        context_response = ollama.embed(
            model = embed_model,
            input = context
        )
        
        # Check if valid embedding vectors were returned
        if 'embeddings' not in input_response or not input_response['embeddings'] or \
           'embeddings' not in context_response or not context_response['embeddings']:
            print("Warning: Ollama did not return valid embedding vectors")
            return 0.5  # Return a moderate default value
        
        input_embedding = np.array(input_response['embeddings'][0])
        context_embedding = np.array(context_response['embeddings'][0])

        similarity_score = cosine_distance(input_embedding, context_embedding)
        return similarity_score
    except Exception as e:
        print(f"Error computing embedding similarity: {str(e)}")
        return 0.5  # Return default value on error
    
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
    Combines the top k text chunks from a dictionary into a single string
    
    Args:
        chunks_dict: Dictionary containing source IDs and corresponding chunk contents {source_id: chunk_content}
        top_k: Number of text chunks to combine, None means combine all
        separator: Separator to use between text chunks
        
    Returns:
        str: Combined text string
    """
    # If dictionary is empty, return empty string
    if not chunks_dict:
        return ""
    
    # Get all text chunks
    chunks_list = list(chunks_dict.items())
    
    # If top_k is specified and valid, limit the number of chunks
    if top_k is not None and top_k > 0:
        chunks_list = chunks_list[:min(top_k, len(chunks_list))]
    
    # Build the combined text
    combined_text = ""
    for i, (source_id, content) in enumerate(chunks_list):
        if i > 0:  # Add separator (except for the first chunk)
            combined_text += separator
        # Optionally add source_id as an identifier
        combined_text += f"[{source_id}] {content}"
    
    return combined_text

def get_query_related_chunks(rag, query, param=None, top_k=None, return_combined_text=False):
    """
    Retrieves all text chunks related to the query
    
    Args:
        rag: LightRAG instance
        query: Query text
        param: Query parameters, creates a new QueryParam when None
        top_k: Number of text chunks to return, None means return all related chunks
        return_combined_text: Whether to return a combined text string instead of a dictionary
        
    Returns:
        dict or str: By default returns a dictionary containing source IDs and corresponding chunk contents {source_id: chunk_content},
                   when return_combined_text=True returns a combined text string
    """
    # Set default query parameters
    if param is None:
        param = QueryParam(mode="hybrid")
    
    # Clone param to avoid modifying the original object
    debug_param = copy.deepcopy(param)
    # Enable DEBUG mode to make LightRAG return internal information
    debug_param.debug = True
    
    # Get query keywords
    loop = asyncio.get_event_loop()
    hl_keywords, ll_keywords = loop.run_until_complete(
        extract_keywords_only(
            text=query,
            param=param,
            global_config=rag.__dict__,
            hashing_kv=rag.llm_response_cache
        )
    )
    
    print(f"Querying Keyword: HL={hl_keywords}, LL={ll_keywords}")
    
    # Collect source_ids of relevant nodes from the knowledge graph
    related_source_ids = set()
    graph = rag.chunk_entity_relation_graph._graph
    
    # Match nodes based on keywords
    for node, data in graph.nodes(data=True):
        node_str = str(node).upper()
        node_desc = str(data.get("description", "")).upper()
        
        # Check if node name or description contains any keywords
        if any(kw.upper() in node_str for kw in hl_keywords + ll_keywords) or \
           any(kw.upper() in node_desc for kw in hl_keywords + ll_keywords):
            
            if "source_id" in data:
                if "<SEP>" in data["source_id"]:
                    for sid in data["source_id"].split("<SEP>"):
                        related_source_ids.add(sid.strip())
                else:
                    related_source_ids.add(data["source_id"].strip())
    
    # Also collect relevant source_ids from edges
    for src, tgt, edge_data in graph.edges(data=True):
        src_str = str(src).upper()
        tgt_str = str(tgt).upper()
        desc = str(edge_data.get("description", "")).upper()
        keywords = str(edge_data.get("keywords", "")).upper()
        
        # Check if edge information contains any keywords
        if any(kw.upper() in src_str + tgt_str + desc + keywords for kw in hl_keywords + ll_keywords):
            if "source_id" in edge_data:
                if "<SEP>" in edge_data["source_id"]:
                    for sid in edge_data["source_id"].split("<SEP>"):
                        related_source_ids.add(sid.strip())
                else:
                    related_source_ids.add(edge_data["source_id"].strip())
    
    # Get text chunks from text_chunks
    result_chunks = {}
    
    # Prioritize retrieving data from client storage (in-memory data)
    if hasattr(rag.text_chunks, "client_storage") and "data" in rag.text_chunks.client_storage:
        chunks = rag.text_chunks.client_storage["data"]
        
        for chunk_id, chunk_data in chunks.items():
            # Direct match with chunk_id
            if chunk_id in related_source_ids:
                result_chunks[chunk_id] = chunk_data["content"]
            # Or match with the source_id field of the chunk
            elif "source_id" in chunk_data and chunk_data["source_id"] in related_source_ids:
                result_chunks[chunk_data["source_id"]] = chunk_data["content"]
    else:
        # If client storage is not available, retrieve from database
        async def get_chunks():
            chunks_data = {}
            for source_id in related_source_ids:
                chunk = await rag.text_chunks.get_by_id(source_id)
                if chunk and "content" in chunk:
                    chunks_data[source_id] = chunk["content"]
            return chunks_data
        
        result_chunks = loop.run_until_complete(get_chunks())
    
    # If the number of returned text chunks needs to be limited
    if top_k is not None and top_k > 0 and len(result_chunks) > top_k:
        # Convert to list and take the first top_k elements
        chunks_items = list(result_chunks.items())[:top_k]
        # Rebuild dictionary
        result_chunks = {k: v for k, v in chunks_items}
    
    # Return combined text string or original dictionary
    if return_combined_text:
        return combine_top_k_chunks(result_chunks, top_k=None)  # Number already limited above, no need to limit again
    
    return result_chunks

def generate_integrated_response(question, response_dict):
    """
    Generate an integrated response using OpenAI's gpt-4o-mini-2024-07-18 model
    
    Args:
        question: The original question
        response_dict: Dictionary containing personality responses and weights {personality: (response, weight)}
        
    Returns:
        str: Integrated response
    """
    # Prepare the prompt text
    prompt_text = f"Question: {question}\n\n"
    
    # Add each personality's response and weight
    for personality, (response, weight) in response_dict.items():
        # Format the personality name with proper capitalization
        formatted_personality = personality.capitalize()
        prompt_text += f"{formatted_personality} Response (Weight: {weight:.4f}):\n{response}\n\n"
    
    # Call OpenAI API
    try:
        # Using the standard OpenAI chat completion API instead of 'responses'
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": integration_prompt},
                {"role": "user", "content": prompt_text}
            ],
        )
        
        # Return the generated text from the message content
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API Error: {str(e)}")
        return f"Failed to generate integrated response: {str(e)}"

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
        "If you could speak to your father Oedipus now, what would you ask him about defying authority?",
        "What frightens you more: an unmarked grave or a life of compromise?",
        "What parts of yourself do you recognize in Creon that you refuse to acknowledge?",
        "If Ismene had joined you, would your sacrifice feel more or less meaningful?",
    ]
    
    # System prompt for each personality
    system_prompt = "Please answer the following question as Antigone with your personality traits. "
    
    # Execute queries for each question
    print("\n=== Generating Answers for All Personalities ===")
    for question in antigone_questions:
        print(f"\nQuestion: {question}")
        print("-" * 80)
        response_dict = {}
        similarity_scores = np.array([])  # Initialize empty numpy array to store similarity scores
        
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
            # Compute Similiarty between genereated answers and text chunks
            similarity_score = compute_embedding_similarity(result, combined_text)
            print(f"Answer-Context Similarity: {similarity_score:.4f}")

            # Store result and similarity score in response_dict
            response_dict[personality] = (result, similarity_score)
            
            # Append similarity score to our array
            similarity_scores = np.append(similarity_scores, similarity_score)
            
            # Print the first 200 characters of combined text for reference
            print(f"Context (excerpt): {combined_text[:200]}..." if len(combined_text) > 200 else combined_text)

        # Print the similarity scores we collected
        print("\nSimilarity scores:", similarity_scores)
        
        # Apply softmax function to convert similarity scores to probabilities
        probabilities = softmax(similarity_scores)
        print("Softmax probabilities:", probabilities)
        
        # Create a mapping from personality to softmax probability
        personality_to_prob = {}
        for i, personality in enumerate([p for p in personalities if p in response_dict]):
            personality_to_prob[personality] = probabilities[i]
        
        # Update similarity scores in response_dict using softmax probabilities
        for personality in response_dict:
            result, _ = response_dict[personality]  # Unpack the tuple, keep only the result
            # Replace original similarity score with softmax probability
            response_dict[personality] = (result, personality_to_prob[personality])
        
        # Print the updated response_dict with softmax probabilities
        print("\nUpdated response dictionary with softmax probabilities:")
        for personality, (result, prob) in response_dict.items():
            print(f"{personality}: prob={prob:.4f}")
            
        # Generate integrated response using OpenAI's gpt-4o-mini-2024-07-18 model
        print("\n=== Generating Integrated Response ===")
        integrated_response = generate_integrated_response(question, response_dict)
        print("\nIntegrated Response:")
        print(integrated_response)

        print("=" * 80)


# If this script is run directly, execute the main function
if __name__ == "__main__":
    main()
