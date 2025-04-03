import os
import json
import torch
import argparse
import glob
import random
from tqdm import tqdm
from torch_geometric.data import Data, Dataset
from src.utils.embedder import load_model, load_text2embedding


def encode_text_attributes(graph, embedder_name, model, tokenizer, device):
    print(f"Encoding text attributes using {embedder_name}...")
    text2embedding = load_text2embedding[embedder_name]

    combined_texts = []
    for i in range(len(graph.title)):
        title = graph.title[i]
        abstract = graph.abstract[i] if graph.abstract[i] != "N/A" else ""
        
        if abstract:
            combined_text = f"{title}. {abstract}"
        else:
            combined_text = title
            
        combined_texts.append(combined_text)
    
    text_embeddings = text2embedding(model, tokenizer, device, combined_texts)
    return text_embeddings

def get_paper_indices(graph, paper_titles):
    indices = []
    title_to_idx = {title.lower().strip(): i for i, title in enumerate(graph.title)}
    
    for title in paper_titles:
        title_lower = title.lower().strip()
        if title_lower in title_to_idx:
            indices.append(title_to_idx[title_lower])
    
    return indices

def extract_subgraph(graph, paper_indices):
    if not paper_indices:
        return None
    
    idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(paper_indices)}
    
    x = graph.x[paper_indices]
    edge_index = graph.edge_index
    mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
    
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        if src in idx_map and dst in idx_map:
            mask[i] = True
    
    subgraph_edges = edge_index[:, mask]
    for i in range(subgraph_edges.size(1)):
        subgraph_edges[0, i] = idx_map[subgraph_edges[0, i].item()]
        subgraph_edges[1, i] = idx_map[subgraph_edges[1, i].item()]
    
    arxiv_ids = [graph.arxiv_id[i] for i in paper_indices]
    titles = [graph.title[i] for i in paper_indices]
    abstracts = [graph.abstract[i] for i in paper_indices]
    paper_ids = paper_indices
    
    subgraph = Data(
        x=x,
        edge_index=subgraph_edges,
        arxiv_id=arxiv_ids,
        title=titles,
        abstract=abstracts,
        paper_id=paper_ids 
    )
    
    return subgraph

def update_graph(citation_graph_path, cached_graphs_dir, embedder_name):
    model, tokenizer, device = load_model[embedder_name]()
    
    citation_graph = torch.load(citation_graph_path)
    text_embeddings = encode_text_attributes(citation_graph, embedder_name, model, tokenizer, device)
    
    citation_graph.x = text_embeddings
    torch.save(citation_graph, citation_graph_path)
    
    title_to_idx = {title.lower().strip(): i for i, title in enumerate(citation_graph.title)}
    
    for filename in os.listdir(cached_graphs_dir):
        if not filename.endswith('.pt'):
            continue
        
        graph_path = os.path.join(cached_graphs_dir, filename)
        graph = torch.load(graph_path)
        
        updated = False
        for i, title in enumerate(graph.title):
            title_lower = title.lower().strip()
            if title_lower in title_to_idx:
                citation_idx = title_to_idx[title_lower]
                graph.x[i] = citation_graph.x[citation_idx]
                updated = True
        
        if updated:
            torch.save(graph, graph_path)
            print(f"Updated cached graph: {filename}")
    
    return title_to_idx

def process_taxonomy(taxonomy_file, cached_graphs_dir, output_dir):
    taxonomy_name = os.path.splitext(os.path.basename(taxonomy_file))[0]
    graph_path = os.path.join(cached_graphs_dir, f"{taxonomy_name}.pt")
    
    if not os.path.exists(graph_path):
        print(f"Cached graph not found for taxonomy {taxonomy_name}")
        return 0
    
    graph = torch.load(graph_path)
    
    with open(taxonomy_file, 'r', encoding='utf-8') as f:
        taxonomy_data = json.load(f)
    
    print(f"Processing taxonomy: {taxonomy_name}")
    
    subgraph_count = 0
    for concept in tqdm(taxonomy_data, desc="Processing concepts"):
        concept_id = concept['id']
        concept_name = concept['concept']
        paper_titles = concept['papers']
        
        if not paper_titles:
            continue
            
        paper_indices = get_paper_indices(graph, paper_titles)
        if not paper_indices:
            continue
            
        subgraph = extract_subgraph(graph, paper_indices)
        if subgraph is None or subgraph.edge_index.size(1) == 0:
            continue
        
        concept_filename = f"{concept_id}.pt"
        concept_path = os.path.join(output_dir, concept_filename)
        torch.save(subgraph, concept_path)
        
        subgraph_count += 1
    
    print(f"Successfully processed {subgraph_count} concept subgraphs for taxonomy {taxonomy_name}")
    return subgraph_count

def process_all_taxonomies(taxonomies_dir, cached_graphs_dir, output_dir, citation_graph_path, embedder_name):
    os.makedirs(output_dir, exist_ok=True)
    
    update_graph(citation_graph_path, cached_graphs_dir, embedder_name)
    
    total_subgraph_count = 0
    for filename in os.listdir(taxonomies_dir):
        if not filename.endswith('.json'):
            continue
            
        taxonomy_file = os.path.join(taxonomies_dir, filename)
        count = process_taxonomy(taxonomy_file, cached_graphs_dir, output_dir)
        total_subgraph_count += count
    
    print(f"Total concept subgraphs processed: {total_subgraph_count}")
    return total_subgraph_count


class ConceptDataset(Dataset):
    def __init__(self, root_dir, concept_graphs_dir, labeled_taxonomies_dir, split='train', 
                 train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, transform=None, pre_transform=None):
        self.root_dir = root_dir
        self.concept_graphs_dir = concept_graphs_dir
        self.labeled_taxonomies_dir = labeled_taxonomies_dir
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1"
        
        super(ConceptDataset, self).__init__(root_dir, transform, pre_transform)
        
        self.concept_files = self.get_split_files()
        self.concept_to_label = self.build_concept_label_mapping()
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(set(self.concept_to_label.values())))}
        
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(len(self.concept_files))]
    
    def build_concept_label_mapping(self):
        concept_to_label = {}
        
        for taxonomy_file in glob.glob(os.path.join(self.labeled_taxonomies_dir, '*.json')):
            with open(taxonomy_file, 'r') as f:
                taxonomy_data = json.load(f)
            
            for concept in taxonomy_data:
                concept_id = concept['id']
                concept_label = concept['concept']
                concept_to_label[concept_id] = concept_label
        
        return concept_to_label
    
    def get_split_files(self):
        taxonomy_to_concepts = {}
        
        for concept_file in glob.glob(os.path.join(self.concept_graphs_dir, '*.pt')):
            concept_id = os.path.splitext(os.path.basename(concept_file))[0]
            if '_' in concept_id:
                taxonomy_id = concept_id.split('_')[0]
                if taxonomy_id not in taxonomy_to_concepts:
                    taxonomy_to_concepts[taxonomy_id] = []
                taxonomy_to_concepts[taxonomy_id].append(concept_file)
        
        taxonomies = list(taxonomy_to_concepts.keys())
        
        random.seed(42)
        random.shuffle(taxonomies)
        
        n_taxonomies = len(taxonomies)
        train_end = int(n_taxonomies * self.train_ratio)
        val_end = train_end + int(n_taxonomies * self.val_ratio)
        
        if self.split == 'train':
            split_taxonomies = taxonomies[:train_end]
        elif self.split == 'val':
            split_taxonomies = taxonomies[train_end:val_end]
        elif self.split == 'test':
            split_taxonomies = taxonomies[val_end:]
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
        concept_files = []
        for taxonomy_id in split_taxonomies:
            concept_files.extend(taxonomy_to_concepts[taxonomy_id])
        
        return concept_files
    
    def process(self):
        for i, concept_file in enumerate(self.concept_files):
            concept_id = os.path.splitext(os.path.basename(concept_file))[0]
            
            graph = torch.load(concept_file)
            
            if concept_id in self.concept_to_label:
                label = self.concept_to_label[concept_id]
                label_idx = self.label_to_idx[label]
            else:
                continue
            
            graph.y = torch.tensor([label_idx], dtype=torch.long)
            graph.concept_id = concept_id
            graph.concept_label = label
            
            if self.pre_transform is not None:
                graph = self.pre_transform(graph)
            
            torch.save(graph, os.path.join(self.processed_dir, f'data_{i}.pt'))
    
    def len(self):
        return len(self.concept_files)
    
    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        
        if self.transform is not None:
            data = self.transform(data)
        
        return data
    
    def get_num_classes(self):
        return len(self.label_to_idx)


def main():
    parser = argparse.ArgumentParser(description='Extract concept subgraphs from labeled taxonomies')
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_dataset_dir = os.path.join(script_dir, '..', '..', 'dataset')
    
    parser.add_argument('--taxonomies_dir', default=os.path.join(default_dataset_dir, 'labeled_taxonomies'),
                        help='Directory containing labeled taxonomies')
    parser.add_argument('--cached_graphs_dir', default=os.path.join(default_dataset_dir, 'cached_graphs'),
                        help='Directory containing cached citation graphs')
    parser.add_argument('--output_dir', default=os.path.join(default_dataset_dir, 'concept_graphs'),
                        help='Directory to save concept subgraphs')
    parser.add_argument('--citation_graph', default=os.path.join(default_dataset_dir, 'citation_graph.pt'),
                        help='Path to the large citation graph')
    parser.add_argument('--embedder', default='sbert', choices=['sbert', 'contriever', 'word2vec'],
                        help='Text embedder to use')
    parser.add_argument('--create_dataset', action='store_true',
                        help='Create train/val/test datasets after processing')
    parser.add_argument('--dataset_root', default=os.path.join(default_dataset_dir, 'processed'),
                        help='Root directory for processed datasets')
    
    args = parser.parse_args()

    process_all_taxonomies(args.taxonomies_dir, args.cached_graphs_dir, args.output_dir, 
                           args.citation_graph, args.embedder)
    
    if args.create_dataset:
        os.makedirs(args.dataset_root, exist_ok=True)
        
        train_dataset = ConceptDataset(
            root_dir=os.path.join(args.dataset_root, 'train'),
            concept_graphs_dir=args.output_dir,
            labeled_taxonomies_dir=args.taxonomies_dir,
            split='train'
        )
        
        val_dataset = ConceptDataset(
            root_dir=os.path.join(args.dataset_root, 'val'),
            concept_graphs_dir=args.output_dir,
            labeled_taxonomies_dir=args.taxonomies_dir,
            split='val'
        )
        
        test_dataset = ConceptDataset(
            root_dir=os.path.join(args.dataset_root, 'test'),
            concept_graphs_dir=args.output_dir,
            labeled_taxonomies_dir=args.taxonomies_dir,
            split='test'
        )
        
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        print(f"Number of classes: {train_dataset.get_num_classes()}")

if __name__ == "__main__":
    main()