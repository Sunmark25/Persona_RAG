#!/usr/bin/env python3
import networkx as nx
import argparse
from pathlib import Path

def extract_subgraph(graph, nodes, max_distance=2):
    """
    Extract a subgraph containing nodes within max_distance of given nodes.
    
    Args:
        graph: NetworkX graph
        nodes: List of node IDs to use as focal points
        max_distance: Maximum distance from focal nodes (default: 2)
    
    Returns:
        NetworkX graph containing the subgraph
    """
    # Create set to store all nodes within range
    nearby_nodes = set()
    
    # For each focal node, get nodes within max_distance
    for node in nodes:
        # Validate node exists
        if node not in graph:
            raise ValueError(f"Node {node} not found in graph")
        
        # Get ego graph (nodes within max_distance)
        ego = nx.ego_graph(graph, node, radius=max_distance)
        nearby_nodes.update(ego.nodes())
    
    # Create subgraph with nearby nodes
    subgraph = graph.subgraph(nearby_nodes).copy()
    
    return subgraph

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Extract subgraph within N edges of specified nodes'
    )
    parser.add_argument('input_file', type=str, help='Input graphml file path')
    parser.add_argument('output_file', type=str, help='Output graphml file path')
    parser.add_argument('nodes', type=str, nargs='+', help='1-2 node IDs to use as focal points')
    parser.add_argument('--distance', type=int, default=2, 
                      help='Maximum distance from focal nodes (default: 2)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate number of nodes
    if len(args.nodes) > 2:
        raise ValueError("Maximum of 2 focal nodes allowed")
    
    # Load input graph
    try:
        graph = nx.read_graphml(args.input_file)
    except Exception as e:
        raise Exception(f"Error reading graphml file: {e}")
    
    # Extract subgraph
    try:
        subgraph = extract_subgraph(graph, args.nodes, args.distance)
    except Exception as e:
        raise Exception(f"Error extracting subgraph: {e}")
    
    # Create output directory if needed
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save subgraph
    try:
        nx.write_graphml(subgraph, args.output_file)
        print(f"Subgraph saved to {args.output_file}")
        print(f"Original graph: {len(graph)} nodes, {len(graph.edges)} edges")
        print(f"Subgraph: {len(subgraph)} nodes, {len(subgraph.edges)} edges")
    except Exception as e:
        raise Exception(f"Error saving graphml file: {e}")

if __name__ == "__main__":
    main()
