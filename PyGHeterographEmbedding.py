import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
import torch_geometric.nn as nn
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear,to_hetero,to_hetero_with_bases
from torch_geometric.utils import subgraph
from torch_geometric.loader import HGTLoader
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import torch
import pandas as pd
import os
import os.path as osp
import sys
import numpy as np
import copy
from tqdm import tqdm



data = HeteroData()



node_feature_index = np.array([[1, 2, 3, 4],[-0.2, 0.3, 0.0, -0.5], [-0.2, 0.3, 0.0, -0.5], [0.2, 1.3, 1.0, -4],[2, 3.3, 2.0, 4.5],[-0.2, 0.3, 0.0, -0.5], [0.2, 1.3, 1.0, -4],[2, 3.3, 2.0, 4.5]],dtype=np.float32)
node_label = np.array([1, 2, 3, 4,0,2,1,3],dtype=int)
edge_index_temporal = np.array([[1, 2, 3, 4,0],[7, 6, 5, 4,7]],dtype=int)
edge_index_spatial = np.array([[4, 2, 3],[2, 6, 7]],dtype=int)
edge_attr_temporal = np.array([[1, 2, 3, 4,0,2,1,3],[7, 6, 5, 4,7,4,5,3],[1, 2, 3, 4,0,2,1,3],[1, 2, 3, 4,0,2,1,3],[1, 2, 3, 4,0,2,1,3]],dtype=np.float32)
edge_attr_spatial = np.array([[1, 22, 3, 4],[7, 6, 5, 4],[0,2,1,3]],dtype=np.float32)



data['object'].x = torch.from_numpy(node_feature_index)
data['object'].id = torch.from_numpy(np.array([0,1,2,3,4,5,6,7]))
data['object'].y = torch.from_numpy(node_label)
data[('object','temporal_link','object')].edge_index = torch.from_numpy(edge_index_temporal)
data[('object','spatial_link','object')].edge_index = torch.from_numpy(edge_index_spatial)
data[('object','temporal_link','object')].edge_attr = torch.from_numpy(edge_attr_temporal)
data[('object','temporal_link','object')].edge_attr = torch.from_numpy(edge_attr_temporal)

train_mask = torch.rand(data.num_nodes) < 0.8
print(train_mask.type)
test_mask = ~train_mask

data['object'].train_mask=train_mask
data['object'].test_mask=test_mask

print(data.metadata())

# from torch_geometric.transforms import RandomLinkSplit

# transform = RandomLinkSplit(is_undirected=False,edge_types=[('object','temporal_link','object'),('object','spatial_link','object')])
# train_data, val_data, test_data = transform(data)



# train_data = copy.copy(data)
# train_data['object'].x = subgraph(train_mask, data['object'].x, relabel_nodes=True)
# train_data['object'].y=subgraph(train_mask, data['object'].y, relabel_nodes=True)
# train_data[('object','temporal_link','object')].edge_index = subgraph(train_mask, data[('object','temporal_link','object')].edge_index, relabel_nodes=True)
# train_data[('object','temporal_link','object')].edge_attr=subgraph(train_mask, data[('object','temporal_link','object')].edge_attr, relabel_nodes=True)
# train_data[('object','temporal_link','object')].edge_index = subgraph(train_mask, data[('object','temporal_link','object')].edge_index, relabel_nodes=True)
# train_data[('object','temporal_link','object')].edge_attr=subgraph(train_mask, data[('object','temporal_link','object')].edge_attr, relabel_nodes=True)

# test_data = copy.copy(data)
# test_data['object'].x = subgraph(test_mask, data['object'].x, relabel_nodes=True)
# test_data['object'].y=subgraph(test_mask, data['object'].y, relabel_nodes=True)
# test_data[('object','temporal_link','object')].edge_index = subgraph(test_mask, data[('object','temporal_link','object')].edge_index, relabel_nodes=True)
# test_data[('object','temporal_link','object')].edge_attr=subgraph(test_mask, data[('object','temporal_link','object')].edge_attr, relabel_nodes=True)
# test_data[('object','temporal_link','object')].edge_index = subgraph(test_mask, data[('object','temporal_link','object')].edge_index, relabel_nodes=True)
# test_data[('object','temporal_link','object')].edge_attr=subgraph(test_mask, data[('object','temporal_link','object')].edge_attr, relabel_nodes=True)

print(data['object'].train_mask)
train_dataloader = HGTLoader(data,batch_size=1,num_samples={key: [4] * 2 for key in data.node_types},input_nodes=('object', data['object'].train_mask))
test_dataloader = HGTLoader(data,batch_size=1,num_samples={key: [8] * 4 for key in data.node_types},input_nodes=('object', data['object'].test_mask))
# for batch in enumerate(train_dataloader):
#     print(batch)
#     print(batch['object'].batch_size)
            
# print(train_dataloader)
# this is the to_hetero_graph method 


class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv((4, 4), 32)
        self.conv2 = SAGEConv((32, 32), 32)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return x

def main():
    
    model = GNN()
    model = to_hetero_with_bases(model, data.metadata(), num_bases=3)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 0 if sys.platform.startswith('win') else 4
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.01)


    def train_loop(dataloader, model,  optimizer):
        size = len(dataloader.dataset)
        for batch in tqdm(dataloader):
            print(batch)
            pred = model(batch.x_dict, batch.edge_index_dict)
            # print(list(pred.values())[0])
            # pred_list = []
            # for item in pred['object']:
            #     item = torch.Tensor(item)
            #     pred_list.append(item)
            # print(torch.tensor(pred_list))
            # print(batch['object'].y)
            loss = F.cross_entropy(list(pred.values())[0], batch['object'].y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.item()
            print(f"loss: {loss:>7f}" )


    def test_loop(dataloader, model):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        print(num_batches)
        test_loss, correct = 0, 0
        
        with torch.no_grad():
            for batch in dataloader:
                pred = model(batch.x_dict, batch.edge_index_dict)
                loss = F.cross_entropy(list(pred.values())[0], batch['object'].y)
                test_loss += loss.item()
                correct += (list(pred.values())[0].numpy().argmax(1) == batch['object'].y.numpy()).sum().item()
        test_loss /= num_batches+0.01
        correct /= size+0.01
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, optimizer)
        test_loop(test_dataloader, model)
    print("Done!")

    # @torch.no_grad()
    # def plot_points(colors):
    #     model.eval()
    #     z = model(torch.arange(data.num_nodes, device=device))
    #     z = TSNE(n_components=2).fit_transform(z.cpu().numpy())
    #     y = data.y.cpu().numpy()

    #     plt.figure(figsize=(8, 8))
    #     for i in range(dataset.num_classes):
    #         plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
    #     plt.axis('off')
    #     plt.show()

    # colors = [
    #     '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535',
    #     '#ffd700'
    # ]
    # plot_points(colors)


if __name__ == "__main__":
    main()