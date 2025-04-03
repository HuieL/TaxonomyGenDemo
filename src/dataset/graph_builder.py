import os
import torch
import argparse
from tqdm import tqdm
import time
import random
from src.utils.data_utils import build_tree, merge_graphs, setup_proxy


def load_papers_from_txt(file_path):
    papers = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split(', ', 1) 
            if len(parts) == 2:
                arxiv_id, title = parts
                papers.append((arxiv_id, title))
            else:
                print(f"Warning: Could not parse line: {line}")
    
    return papers

def build_and_merge_citation_graph(papers, dataset_dir, max_papers=None):
    setup_proxy()

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    cache_dir = os.path.join(dataset_dir, "cached_graphs")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    if max_papers is not None and max_papers > 0:
        papers = papers[:max_papers]

    paper_graphs = []
    
    for i, (arxiv_id, title) in enumerate(tqdm(papers, desc="Building citation trees")):
        cache_file = os.path.join(cache_dir, f"{arxiv_id}.pt")
        
        # Check if the graph is already cached
        if os.path.exists(cache_file):
            try:
                paper_graph = torch.load(cache_file)
                paper_graphs.append(paper_graph)
                print(f"Loaded cached graph for {arxiv_id}: {title}")
                continue
            except Exception as e:
                print(f"Error loading cached graph for {arxiv_id}: {e}")
        
        try:
            paper_graph = build_tree(arxiv_id, title)
            torch.save(paper_graph, cache_file)
            paper_graphs.append(paper_graph)
            print(f"Built graph for {arxiv_id}: {title}")
            time.sleep(random.uniform(1, 3))
            
        except Exception as e:
            print(f"Error building graph for {arxiv_id}: {e}")
    
    if paper_graphs:
        try:
            print("Merging all graphs...")
            merged_graph = merge_graphs(paper_graphs)
            
            merged_path = os.path.join(dataset_dir, "citation_graph.pt")
            torch.save(merged_graph, merged_path)
            print(f"Saved merged graph to {merged_path}")
            
            return merged_graph
        except Exception as e:
            print(f"Error merging graphs: {e}")
            return None
    else:
        print("No graphs to merge.")
        return None

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "..", "..", "dataset")
    input_file = os.path.join(dataset_dir, "root_papers.txt")
    
    parser = argparse.ArgumentParser(description="Build and merge citation graphs from a list of papers.")
    parser.add_argument("--input-file", default=input_file, 
                        help=f"Path to the text file containing paper information")
    parser.add_argument("--dataset-dir", default=dataset_dir, 
                        help=f"Directory where dataset files are stored")
    parser.add_argument("--max-papers", type=int, default=None, 
                        help="Maximum number of papers to process")
    args = parser.parse_args()
    
    papers = load_papers_from_txt(args.input_file)
    build_and_merge_citation_graph(papers, args.dataset_dir, args.max_papers)

if __name__ == "__main__":
    main()