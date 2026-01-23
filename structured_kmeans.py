import torch
import torch.nn as nn
import torch.nn.functional as F


# This module can be used as a general clustering model
# It enforces a certain topology on the cluster centers (manifold-learning) 
# Mostly used as a refinement step after some initial clustering, as it is sensitive to the initial cluster centers
class StructuredKMeans(nn.Module):
    def __init__(self, initial_centers, n_clusters, n_features, connections=None, device='cpu', k=0.1, T=0.2):
        """
        Initialize the structured k-means model

        Args:
            initial_centers (torch.Tensor): initial cluster centers, shape [n_clusters, n_features]
            connections (torch.Tensor): connections between clusters, shape [n_clusters, n_clusters]
            n_clusters (int): number of clusters
            n_features (int): number of features
            device (torch.device): device to run the model
            k (float): weight of the cluster structure loss
            T (float): temperature that controls the manifold shape, lower T -> more uniform (learns more global structure)
        """
        super(StructuredKMeans, self).__init__()
        if initial_centers is not None:
            self.centers = nn.Parameter(torch.from_numpy(initial_centers).to(device))
        else:
            self.centers = nn.Parameter(torch.zeros((n_clusters, n_features), device=device))
        self.connections = connections
        self.diag = torch.diag(torch.ones(n_clusters, device=device) * 1e4)
        self.k = k
        self.T = T

    def n_clusters(self):
        return self.centers.size(0)
    
    def n_features(self):
        return self.centers.size(1)

    def forward(self, x):
        """
        Calculate the distance from input data to each cluster center

        Args:
            x (torch.Tensor): input data, shape [batch_size, n_features]

        Returns:
            torch.Tensor: distance from each sample to each cluster center, shape [batch_size, n_clusters]
        """
        # pca_features: (Batch, n_features)
        # pca_centroid: (n_clusters, n_features)
        distances = torch.cdist(x, self.centers, p=2) ** 2  # (Batch, n_clusters) 
        # use softmax to calculate the probability
        # # use -distances * 4 to make the probability more concentrated
        prob = F.softmin(distances, dim=1)  # (Batch, n_clusters)
        return distances, prob

    def compute_loss(self, x):
        """
        Calculate the total loss, including the distance from each sample to the nearest cluster center and the structure loss between clusters

        Args:
            x (torch.Tensor): input data, shape [batch_size, n_features]

        Returns:
            torch.Tensor: total loss value
        """
        # calculate the distance from each sample to each cluster center
        distances, _ = self.forward(x)

        # calculate the loss of the distance from each sample to the nearest cluster center
        min_distances = torch.min(distances, dim=1)[0]
        # print(min_distances.shape)
        sample_mean_distance = torch.mean(min_distances)
        # sample_std_distance = torch.std(min_distances)
        sample_loss = sample_mean_distance

        # calculate the structure loss between clusters
        # calculate the distance between each cluster center
        center_distances = torch.cdist(self.centers, self.centers, p=2)
        # set the diagonal elements to infinity to avoid calculating the distance between the same cluster center
        
        if self.connections is None:
            center_distances = center_distances + self.diag

            # use softmax to calculate the weight of the distance between each cluster center
            weights = F.softmin(center_distances /
                                torch.mean(center_distances) * self.T, dim=1)
            weighted_distances = torch.sum(weights * center_distances, dim=1)
        else:
            weights = self.connections
            weighted_distances = torch.mean(weights * center_distances, dim=1)

        # calculate the mean and standard deviation of the weighted distance
        mean_distance = torch.mean(weighted_distances)
        std_distance = torch.std(weighted_distances)

        # calculate the structure loss
        structure_loss = mean_distance * std_distance

        # total loss
        total_loss = sample_loss + self.k * structure_loss

        return total_loss, sample_loss, mean_distance, std_distance


# if __name__ == "__main__":
#     from scipy.io import loadmat, savemat
#     import numpy as np
#     from tqdm import tqdm
#     np.set_printoptions(precision=4, suppress=True)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     n_epochs = 1000
#     f = loadmat("SOM_init.mat")
#     connections = f["connections"]
#     connections = torch.from_numpy(connections).bool().to(device)
#     data = f["norm_data"]
#     data = torch.from_numpy(data).float().to(device)
#     # weights = f["sample_weights"].squeeze()
#     # weights = torch.from_numpy(weights).float().to(device)
#     initial_centers = f["centroids"]
#     n_clusters = initial_centers.shape[0]
#     n_features = initial_centers.shape[1]
#     cluster = StructuredKMeans(initial_centers, n_clusters, n_features, connections=connections, k=1, device=device)
#     cluster = cluster.to(device)
#     cluster = torch.compile(cluster)
#     loss, sample_loss, mean_distance, std_distance = cluster.compute_loss(data)

#     print("Before Training")
#     print(f'Total Loss: {loss.item()}, Sample Loss: {sample_loss.item()}, Mean Distance: {mean_distance.item()}, Std Distance: {std_distance.item()}')

#     optimizer = torch.optim.Adam(cluster.parameters(), lr=0.01)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0.0001)
#     for epoch in tqdm(range(n_epochs), desc="Training"):
#         optimizer.zero_grad()
#         loss, sample_loss, mean_distance, std_distance = cluster.compute_loss(data)
#         loss.backward(retain_graph=True)
#         optimizer.step()
#         scheduler.step()
#         # print(f'Epoch {epoch+1} / {n_epochs}')
#         # print(f'Total Loss: {loss.item()}, Sample Loss: {sample_loss.item()}, Mean Distance: {mean_distance.item()}, Std Distance: {std_distance.item()}')

#     print("After Training")
#     print(f'Total Loss: {loss.item()}, Sample Loss: {sample_loss.item()}, Mean Distance: {mean_distance.item()}, Std Distance: {std_distance.item()}')

#     savemat("SOM_trained.mat", {"centroids": cluster.centers.detach().cpu().numpy()})