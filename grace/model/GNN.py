import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv, TransformerConv, RGATConv

import grace

class GNN(nn.Module):
    def __init__(self, g_dim, h1_dim, h2_dim, num_relations, num_modals, args):
        super(GNN, self).__init__()
        self.args = args

        self.num_modals = num_modals
        
        if args.gcn_conv == "rgcn":
            print("GNN --> Use RGCN")
            self.conv1 = RGCNConv(g_dim, h1_dim, num_relations)
            # self.conv1 = RGATConv(g_dim, h1_dim, num_relations, dropout=0.1)
            self.bn1 = nn.BatchNorm1d(h1_dim)
            self.conv1_1 = RGATConv(g_dim, h1_dim, num_relations)
            self.bn1_1 = nn.BatchNorm1d(h1_dim)

        if args.use_graph_transformer:
            print("GNN --> Use Graph Transformer")
           
            in_dim = h1_dim
            self.conv2_1 = TransformerConv(in_dim, h2_dim)
            self.conv2 = TransformerConv(in_dim, h2_dim, heads=args.graph_transformer_nheads, concat=True)
            self.bn = nn.BatchNorm1d(h2_dim * args.graph_transformer_nheads)
            

    def forward(self, node_features, node_type, edge_index, edge_type):

        if self.args.gcn_conv == "rgcn":
            residual = nn.functional.leaky_relu(self.bn1(self.conv1(node_features, edge_index, edge_type)))
            x = residual + nn.functional.leaky_relu(self.bn1_1(self.conv1_1(residual, edge_index, edge_type)))
            # residual = node_features
            # x = nn.functional.leaky_relu(self.bn1(residual + self.conv1(node_features, edge_index, edge_type)))
            # residual = x
            # x = nn.functional.leaky_relu(self.bn1_1(residual + self.conv1_1(x, edge_index, edge_type)))
        
        if self.args.use_graph_transformer:
            x = self.conv2_1(x, edge_index)
            x = nn.functional.leaky_relu(self.bn(self.conv2(x, edge_index)))
        
        return x
        
