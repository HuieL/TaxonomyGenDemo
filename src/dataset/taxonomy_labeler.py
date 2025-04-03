import os
import json
import re
import argparse


def parse_taxonomy_tree(tree_content):
    lines = tree_content.strip().split('\n')
    root_concept = lines[0].strip()
    
    taxonomy = {
        "concept": root_concept,
        "level": 0,
        "children": []
    }
    
    current_path = [taxonomy]
    pattern = re.compile(r'^(├──|└──|│   |    )*\s*(.+)$')
    
    for line in lines[1:]:
        indent_match = pattern.match(line)
        if not indent_match:
            continue
            
        indent_str = indent_match.group(1) or ""
        concept = indent_match.group(2).strip()
        
        level = (len(indent_str) + 3) // 4
        
        while len(current_path) > level:
            current_path.pop()
        
        new_node = {
            "concept": concept,
            "level": level,
            "children": []
        }
        
        current_path[-1]["children"].append(new_node)
        current_path.append(new_node)
    
    return taxonomy

def load_cluster_data(cluster_dir, taxonomy_name):
    concept_papers = {}
    cluster_file = os.path.join(cluster_dir, f"{taxonomy_name}.json")
    
    if os.path.exists(cluster_file):
        with open(cluster_file, 'r', encoding='utf-8') as f:
            cluster_data = json.load(f)
            
        for concept, papers in cluster_data.items():
            normalized_concept = concept.strip()
            concept_papers[normalized_concept] = papers
    
    return concept_papers

def associate_papers_with_taxonomy(taxonomy, concept_papers):
    def process_node(node):
        concept = node["concept"]
        node["papers"] = concept_papers.get(concept, [])
        
        for child in node["children"]:
            process_node(child)
    
    process_node(taxonomy)
    return taxonomy

def flatten_taxonomy(taxonomy, taxonomy_name):
    flat_list = []
    id_counter = 0
    
    def traverse(node):
        nonlocal id_counter
        concept_id = f"{taxonomy_name}_{id_counter}"
        id_counter += 1
        
        flat_list.append({
            "concept": node["concept"],
            "level": node["level"],
            "papers": node.get("papers", []),
            "id": concept_id
        })
        
        for child in node["children"]:
            traverse(child)
    
    traverse(taxonomy)
    return flat_list

def process_taxonomy_file(taxonomy_file, cluster_dir, output_dir):
    taxonomy_name = os.path.splitext(os.path.basename(taxonomy_file))[0]
    
    with open(taxonomy_file, 'r', encoding='utf-8') as f:
        tree_content = f.read()
    
    taxonomy = parse_taxonomy_tree(tree_content)
    concept_papers = load_cluster_data(cluster_dir, taxonomy_name)
    taxonomy_with_papers = associate_papers_with_taxonomy(taxonomy, concept_papers)
    flat_taxonomy = flatten_taxonomy(taxonomy_with_papers, taxonomy_name)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, f"{taxonomy_name}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(flat_taxonomy, f, indent=2, ensure_ascii=False)
    
    return True

def process_all_taxonomies(taxonomy_dir, cluster_dir, output_dir):
    success_count = 0
    
    for file_name in os.listdir(taxonomy_dir):
        file_path = os.path.join(taxonomy_dir, file_name)
        
        if os.path.isdir(file_path) or not file_name.endswith(('.txt', '.tree')):
            continue
        
        if process_taxonomy_file(file_path, cluster_dir, output_dir):
            success_count += 1
    
    return success_count

def main():
    parser = argparse.ArgumentParser(description='Process taxonomy trees and generate labeled JSON files')
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_dataset_dir = os.path.join(script_dir, '..', '..', 'dataset')
    default_taxonomy_dir = os.path.join(default_dataset_dir, 'taxonomies')
    default_cluster_dir = os.path.join(default_dataset_dir, 'clusters')
    default_output_dir = os.path.join(default_dataset_dir, 'labeled_taxonomies')
    
    parser.add_argument('--taxonomy-dir', default=default_taxonomy_dir,
                        help='Directory containing taxonomy tree files')
    parser.add_argument('--cluster-dir', default=default_cluster_dir,
                        help='Directory containing cluster JSON files')
    parser.add_argument('--output-dir', default=default_output_dir,
                        help='Directory to save output JSON files')
    
    args = parser.parse_args()
    process_all_taxonomies(args.taxonomy_dir, args.cluster_dir, args.output_dir)

if __name__ == "__main__":
    main()