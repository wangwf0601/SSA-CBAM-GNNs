import torch
from torch_geometric.data import Data
from ssa import sparrow_search_algorithm
from cnn import CNN
from gnn import SimpleGNN
from cbam import CBAM
from torchvision import transforms


def evaluate_model(model, dataloader, device):
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        population_size, dimension, max_iterations, producer_ratio, safe_threshold, hazard_awareness = 100, 10, 100, 0.2, 0.5, 0.1
        cnn = CNN()
        cbam = CBAM(in_channels=128)

        for batch in dataloader:
            images, labels = batch['image'].to(device), batch['label'].to(device)

            cnn_output = cnn(images)
            ssa_features, _ = sparrow_search_algorithm(population_size, dimension, max_iterations, producer_ratio,
                                                       safe_threshold, hazard_awareness)
            combined_features = torch.cat((cnn_output, torch.Tensor(ssa_features)), dim=1)
            attention_features = cbam(combined_features)  # 请替换为你的注意力机制模块

            edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)  # 请替换为实际的边索引
            graph_data = Data(x=attention_features, edge_index=edge_index)

            gnn_output = model(graph_data.x, graph_data.edge_index)

            predictions = torch.argmax(gnn_output, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_predictions, all_labels
