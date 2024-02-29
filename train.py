import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch_geometric.data import Data
from ssa import sparrow_search_algorithm
from cnn import CNN
from gnn import SimpleGNN
from cbam import CBAM
from PIL import Image
from torchvision import transforms
import numpy as np
from evaluate import evaluate_model

def image_loading_function(file_path):
    image = Image.open(file_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = transform(image)
    return image
class CustomDataset(Dataset):
    def __init__(self, root_folder):
        self.root_folder = root_folder
        self.classes = sorted(os.listdir(root_folder))
        self.class_to_index = {cls: i for i, cls in enumerate(self.classes)}
        self.data = self.load_data()

    def load_data(self):
        data = []
        for class_name in self.classes:
            class_folder = os.path.join(self.root_folder, class_name)
            for file_path in glob.glob(os.path.join(class_folder, '*.jpg')):
                data.append({'file_path': file_path, 'label': self.class_to_index[class_name]})
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = image_loading_function(sample['file_path'])
        label = sample['label']
        return {'image': image, 'label': label}


if __name__ == '__main__':
    data_folder = 'path/to/your/data/folder'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = CustomDataset(root_folder=data_folder)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    cnn_model = CNN()
    gnn_model = SimpleGNN(num_features=64, hidden_size=64, num_classes=5)
    cbam_module = CBAM(in_channels=128)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(cnn_model.parameters()) + list(gnn_model.parameters()) + list(cbam_module.parameters()),
                           lr=0.001)
    population_size, dimension, max_iterations, producer_ratio, safe_threshold, hazard_awareness = 100, 10, 100, 0.2, 0.5, 0.1

    for epoch in range(40):
        for batch in dataloader:
            images, labels = batch['image'].to(device), batch['label'].to(device)

            cnn_output = cnn_model(images)

            ssa_features, _ = sparrow_search_algorithm(population_size, dimension, max_iterations, producer_ratio,
                                                       safe_threshold, hazard_awareness)

            combined_features = torch.cat((cnn_output, torch.Tensor(ssa_features)), dim=1)

            attention_features = cbam_module(combined_features)

            edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
            graph_data = Data(x=attention_features, edge_index=edge_index)

            gnn_output = gnn_model(graph_data.x, graph_data.edge_index)

            loss = criterion(gnn_output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print("Training completed!")


    # Evaluate the model
    all_predictions, all_labels = evaluate_model(gnn_model, dataloader, device)

    unique_classes, class_counts = np.unique(all_predictions, return_counts=True)

    class_counts_dict = dict(zip(unique_classes, class_counts))
    total_samples = len(all_predictions)
    class_ratios = {cls: count / total_samples for cls, count in class_counts_dict
                    .items()}
    print("Predicted Classes:", all_predictions)
    print("Class Counts:", class_counts_dict)
    print("Class Ratios:", class_ratios)
    print("Training completed!")


