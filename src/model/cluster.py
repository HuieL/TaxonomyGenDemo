import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np
import argparse

from src.dataset.load_data import ConceptDataset
from src.model.gnn import GAT, GCN, GraphTransformer


class HierarchicalClusterModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.2, gnn_type='gat', num_heads=4):
        super(HierarchicalClusterModel, self).__init__()
        self.gnn_type = gnn_type
        self.num_layers = num_layers

        if gnn_type == 'gat':
            self.gnn = GAT(in_channels, hidden_channels, hidden_channels, num_layers, dropout, num_heads)
        elif gnn_type == 'gcn':
            self.gnn = GCN(in_channels, hidden_channels, hidden_channels, num_layers, dropout)
        elif gnn_type == 'gt':
            self.gnn = GraphTransformer(in_channels, hidden_channels, hidden_channels, num_layers, dropout, num_heads)
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
        
        # MLP for calculating same-cluster probability
        self.same_cluster_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, edge_attr=None):
        # Apply GNN to get node embeddings
        node_embeddings, _ = self.gnn(x, edge_index, edge_attr)
        return node_embeddings
    
    def calculate_same_cluster_prob(self, node_embeddings, pair_indices):
        node_i = node_embeddings[pair_indices[0]]
        node_j = node_embeddings[pair_indices[1]]
        
        pair_embeddings = torch.cat([node_i, node_j], dim=1)
        prob = self.same_cluster_mlp(pair_embeddings)
        return prob.squeeze()
    
    def calculate_node_density(self, node_embeddings, edge_index):

        num_nodes = node_embeddings.size(0)
        densities = torch.zeros(num_nodes, device=node_embeddings.device)
        
        # For each node, calculate its density
        for i in range(num_nodes):
            neighbors = edge_index[1][edge_index[0] == i]
            
            if len(neighbors) == 0:
                continue

            neighbor_embeds = node_embeddings[neighbors]
            node_embed = node_embeddings[i]

            cos_sim = F.cosine_similarity(
                node_embed.unsqueeze(0).expand(len(neighbors), -1),
                neighbor_embeds
            )

            pair_indices = torch.stack([
                torch.full((len(neighbors),), i, device=node_embeddings.device),
                neighbors
            ])
            
            same_cluster_probs = self.calculate_same_cluster_prob(node_embeddings, pair_indices)
            densities[i] = (same_cluster_probs * cos_sim).mean()
        
        return densities

    def generate_clusters(self, node_embeddings, edge_index, level=1, p_threshold=0.5):
        num_nodes = node_embeddings.size(0)

        node_densities = self.calculate_node_density(node_embeddings, edge_index)
        if level == 1:
            # Soft clustering for level 1 (allowing nodes to belong to multiple clusters)
            candidate_clusters = []
            for i in range(num_nodes):
                cluster = [i]
                neighbors = edge_index[1][edge_index[0] == i].tolist()

                for j in neighbors:
                    # Check density and same-cluster probability
                    if node_densities[i] < node_densities[j]:
                        pair_indices = torch.tensor([[i], [j]], device=node_embeddings.device)
                        prob = self.calculate_same_cluster_prob(node_embeddings, pair_indices).item()
                        
                        if prob > p_threshold:
                            cluster.append(j)
                
                if len(cluster) > 1:  # Only consider non-singleton clusters
                    candidate_clusters.append(set(cluster))
            
            # Merge clusters that are subsets of others
            final_clusters = []
            for i, cluster_i in enumerate(candidate_clusters):
                is_subset = False
                for j, cluster_j in enumerate(candidate_clusters):
                    if i != j and cluster_i.issubset(cluster_j):
                        is_subset = True
                        break
                
                if not is_subset and len(cluster_i) > 1:
                    final_clusters.append(list(cluster_i))
            
            return final_clusters
        
        else:
            # Hard clustering for higher levels
            connected_edges = []
            for i in range(edge_index.size(1)):
                src, dst = edge_index[0, i].item(), edge_index[1, i].item()

                pair_indices = torch.tensor([[src], [dst]], device=node_embeddings.device)
                prob = self.calculate_same_cluster_prob(node_embeddings, pair_indices).item()

                if prob > p_threshold:
                    connected_edges.append((src, dst))
            
            adj_list = {i: [] for i in range(num_nodes)}
            for src, dst in connected_edges:
                adj_list[src].append(dst)
                adj_list[dst].append(src)  
            
            visited = [False] * num_nodes
            clusters = []
            
            for i in range(num_nodes):
                if not visited[i]:
                    cluster = []
                    self._dfs(i, adj_list, visited, cluster)
                    
                    if len(cluster) > 1:  # Only consider non-singleton clusters
                        clusters.append(cluster)
            
            return clusters
    
    def _dfs(self, node, adj_list, visited, cluster):
        visited[node] = True
        cluster.append(node)
        
        for neighbor in adj_list[node]:
            if not visited[neighbor]:
                self._dfs(neighbor, adj_list, visited, cluster)


class HierarchicalClusterer:
    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.2, 
                 gnn_type='gat', num_heads=4, lr=0.001, weight_decay=5e-4,
                 device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.model = HierarchicalClusterModel(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            gnn_type=gnn_type,
            num_heads=num_heads
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
    
    def train(self, data, labels, epochs=200, p_threshold=0.5, patience=10):
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        edge_attr = data.edge_attr.to(self.device) if hasattr(data, 'edge_attr') else None
        labels = labels.to(self.device)
        
        pairs, targets = self._create_training_pairs(data, labels)
        pairs = pairs.to(self.device)
        targets = targets.to(self.device)
        
        best_loss = float('inf')
        patience_counter = 0
        losses = []
        
        self.model.train()
        for epoch in tqdm(range(epochs), desc="Training"):
            self.optimizer.zero_grad()

            node_embeddings = self.model(x, edge_index, edge_attr)
            probs = self.model.calculate_same_cluster_prob(node_embeddings, pairs)
            loss = F.binary_cross_entropy(probs, targets)

            if hasattr(data, 'hierarchical_labels'):
                hier_loss = self._hierarchical_contrastive_loss(
                    node_embeddings, data.hierarchical_labels, data.edge_index
                )
                loss += hier_loss
            
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        return losses
    
    def _create_training_pairs(self, data, labels):
        num_nodes = data.x.size(0)
        edge_index = data.edge_index.cpu()

        pairs = []
        targets = []
        
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            pairs.append([src, dst])

            same_cluster = (labels[src] == labels[dst]).item()
            targets.append(float(same_cluster))

        pairs = torch.tensor(pairs, dtype=torch.long).t()  
        targets = torch.tensor(targets, dtype=torch.float)  
        
        return pairs, targets
    
    def _hierarchical_contrastive_loss(self, node_embeddings, hierarchical_labels, edge_index, temperature=0.1):
        num_levels = len(hierarchical_labels)
        num_nodes = node_embeddings.size(0)
        loss = 0.0
        
        for level, labels in enumerate(hierarchical_labels):
            level_weight = 1.0 / (level + 1)
            
            for i in range(num_nodes):
                positives = torch.where(labels == labels[i])[0]
                positives = positives[positives != i] 
                
                if len(positives) == 0:
                    continue

                node_embed = node_embeddings[i]
                pos_embeds = node_embeddings[positives]

                pos_sim = F.cosine_similarity(
                    node_embed.unsqueeze(0).expand(len(positives), -1),
                    pos_embeds
                )
                
                negatives = torch.ones(num_nodes, dtype=torch.bool, device=node_embeddings.device)
                negatives[i] = False
                negatives[positives] = False
                
                if not torch.any(negatives):
                    continue
                
                neg_embeds = node_embeddings[negatives]
                neg_sim = F.cosine_similarity(
                    node_embed.unsqueeze(0).expand(neg_embeds.size(0), -1),
                    neg_embeds
                )
                
                pos_exp = torch.exp(pos_sim / temperature)
                neg_exp = torch.exp(neg_sim / temperature)
                
                node_loss = -torch.log(
                    pos_exp.sum() / (pos_exp.sum() + neg_exp.sum() + 1e-8)
                )
                
                loss += level_weight * node_loss
        
        return loss / num_levels
    
    def cluster(self, data, num_levels=2, p_threshold=0.5):
        self.model.eval()
        hierarchical_clusters = []
        node_embeddings_per_level = []

        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        edge_attr = data.edge_attr.to(self.device) if hasattr(data, 'edge_attr') else None
        
        current_x = x
        current_edge_index = edge_index
        current_edge_attr = edge_attr
        
        with torch.no_grad():
            for level in range(1, num_levels + 1):
                node_embeddings = self.model(current_x, current_edge_index, current_edge_attr)
                node_embeddings_per_level.append(node_embeddings)
                
                # Generate clusters for current level
                clusters = self.model.generate_clusters(
                    node_embeddings, current_edge_index, level=level, p_threshold=p_threshold
                )
                
                hierarchical_clusters.append(clusters)
                
                if level < num_levels:
                    # Prepare for next level: create hyper-nodes and hyper-edges
                    next_x, next_edge_index, next_edge_attr = self._create_next_level_graph(
                        current_x, node_embeddings, clusters
                    )
                    
                    current_x = next_x
                    current_edge_index = next_edge_index
                    current_edge_attr = next_edge_attr
        
        return hierarchical_clusters, node_embeddings_per_level
    
    def _create_next_level_graph(self, x, node_embeddings, clusters):
        num_clusters = len(clusters)
        next_x = torch.zeros(num_clusters, node_embeddings.size(1), device=self.device)
        
        # For each cluster, aggregate node features
        for i, cluster in enumerate(clusters):
            # Get the highest density node in the cluster
            cluster_embeds = node_embeddings[cluster]
            cluster_x = x[cluster]
            
            next_x[i] = cluster_embeds.mean(dim=0)
        
        edges = []
        node_to_cluster = {}
        for i, cluster in enumerate(clusters):
            for node in cluster:
                if node not in node_to_cluster:
                    node_to_cluster[node] = []
                node_to_cluster[node].append(i)
        
        # Connect clusters that share nodes or have edges between them
        for i in range(num_clusters):
            for j in range(i + 1, num_clusters):
                # Check if clusters i and j have edges between them
                cluster_i = set(clusters[i])
                cluster_j = set(clusters[j])
                
                if cluster_i.intersection(cluster_j):
                    edges.append((i, j))
                    edges.append((j, i))  

        if edges:
            next_edge_index = torch.tensor(edges, dtype=torch.long, device=self.device).t()
        else:
            next_edge_index = torch.zeros(2, 0, dtype=torch.long, device=self.device)
        
        return next_x, next_edge_index, None


def train_hierarchical_clusterer(dataset, in_channels, hidden_channels, num_layers=2, 
                               dropout=0.2, gnn_type='gat', num_heads=4, lr=0.001, 
                               weight_decay=5e-4, epochs=200, patience=10,
                               p_threshold=0.5, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize clusterer
    clusterer = HierarchicalClusterer(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
        gnn_type=gnn_type,
        num_heads=num_heads,
        lr=lr,
        weight_decay=weight_decay,
        device=device
    )
    
    data = dataset[0] 
    losses = clusterer.train(
        data=data,
        labels=data.y,
        epochs=epochs,
        p_threshold=p_threshold,
        patience=patience
    )
    
    return clusterer, losses

def test_hierarchical_clusterer(clusterer, data, num_levels=2, p_threshold=0.5):
    hierarchical_clusters, node_embeddings = clusterer.cluster(
        data=data,
        num_levels=num_levels,
        p_threshold=p_threshold
    )
    
    return hierarchical_clusters, node_embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and test hierarchical clustering')

    parser.add_argument('--dataset_root', type=str, default='./dataset/processed')
    parser.add_argument('--concept_graphs_dir', type=str, default='./dataset/concept_graphs')
    parser.add_argument('--labeled_taxonomies_dir', type=str, default='./dataset/labeled_taxonomies')
    
    parser.add_argument('--hidden_channels', type=int, default=1024)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--gnn_type', type=str, default='gat', choices=['gat', 'gcn', 'gt'])
    parser.add_argument('--num_heads', type=int, default=4)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--p_threshold', type=float, default=0.5)
    parser.add_argument('--num_levels', type=int, default=2)
    
    args = parser.parse_args()
    
    # Load dataset
    train_dataset = ConceptDataset(
        root_dir=os.path.join(args.dataset_root, 'train'),
        concept_graphs_dir=args.concept_graphs_dir,
        labeled_taxonomies_dir=args.labeled_taxonomies_dir,
        split='train'
    )
    
    data = train_dataset[0]
    in_channels = data.x.size(1)
    
    # Train model
    clusterer, losses = train_hierarchical_clusterer(
        dataset=train_dataset,
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        gnn_type=args.gnn_type,
        num_heads=args.num_heads,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        p_threshold=args.p_threshold
    )
    
    # Test model
    test_dataset = ConceptDataset(
        root_dir=os.path.join(args.dataset_root, 'test'),
        concept_graphs_dir=args.concept_graphs_dir,
        labeled_taxonomies_dir=args.labeled_taxonomies_dir,
        split='test'
    )
    
    test_data = test_dataset[0]
    hierarchical_clusters, node_embeddings = test_hierarchical_clusterer(
        clusterer=clusterer,
        data=test_data,
        num_levels=args.num_levels,
        p_threshold=args.p_threshold
    )
    
    for level, clusters in enumerate(hierarchical_clusters):
        print(f"Level {level+1} clusters: {len(clusters)}")
        print(f"Average cluster size: {sum(len(c) for c in clusters) / len(clusters) if clusters else 0}")