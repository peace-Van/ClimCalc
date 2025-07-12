import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import orthogonal, weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
import torch.nn.functional as F
from structured_kmeans import StructuredKMeans


class MyActivationLayer(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.temp_mu = torch.nn.Parameter(torch.tensor([-18, 0, 10, 22], dtype=torch.float32, device=self.device))
        self.temp_sigma = torch.nn.Parameter(torch.tensor(6, dtype=torch.float32, device=self.device))
        self.precip_mu = torch.nn.Parameter(torch.tensor([10, 40, 100], dtype=torch.float32, device=self.device))
        self.precip_sigma = torch.nn.Parameter(torch.tensor([10, 20, 50], dtype=torch.float32, device=self.device))
        self.precip_mu_log = torch.nn.Parameter(torch.tensor([2, 4, 5], dtype=torch.float32, device=self.device))
        self.precip_sigma_log = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32, device=self.device))

    def forward(self, x):
        low_temp = x[:, 0, :]
        precip = x[:, 1, :]
        high_temp = x[:, 2, :]
        log_precip = torch.log(precip + 1)

        results = torch.zeros((x.shape[0], 2, 9, 12), dtype=torch.float32, device=self.device)
        results[:, 0, 0, :] = (low_temp - self.temp_mu[0]) / self.temp_sigma
        results[:, 0, 1, :] = (precip - self.precip_mu[0]) / self.precip_sigma[0]
        results[:, 0, 2, :] = (high_temp - self.temp_mu[1]) / self.temp_sigma
        results[:, 0, 3, :] = (low_temp - self.temp_mu[1]) / self.temp_sigma
        results[:, 0, 4, :] = (precip - self.precip_mu[1]) / self.precip_sigma[1]
        results[:, 0, 5, :] = (high_temp - self.temp_mu[2]) / self.temp_sigma
        results[:, 0, 6, :] = (low_temp - self.temp_mu[2]) / self.temp_sigma
        results[:, 0, 7, :] = (precip - self.precip_mu[2]) / self.precip_sigma[2]
        results[:, 0, 8, :] = (high_temp - self.temp_mu[3]) / self.temp_sigma

        results[:, 1, :, :] = results[:, 0, :, :]
        results[:, 1, 1, :] = (log_precip - self.precip_mu_log[0]) / self.precip_sigma_log
        results[:, 1, 4, :] = (log_precip - self.precip_mu_log[1]) / self.precip_sigma_log
        results[:, 1, 7, :] = (log_precip - self.precip_mu_log[2]) / self.precip_sigma_log

        return torch.tanh(results)


# class DualGroupNormWithScale(nn.Module):
#     def __init__(self, normalized_shape, device, eps=1e-5):
#         super().__init__()
#         self.norm = nn.GroupNorm(num_groups=2, num_channels=normalized_shape, eps=eps, affine=False)
#         self.scale = nn.Parameter(torch.ones(normalized_shape, dtype=torch.float32, device=device))

#     def forward(self, x):
#         return self.scale * self.norm(x)


class BatchNormWithScale(nn.Module):
    def __init__(self, normalized_shape, device, eps=1e-5):
        super().__init__()
        self.norm = nn.BatchNorm1d(num_features=normalized_shape, eps=eps, affine=False)
        self.scale = nn.Parameter(torch.ones(normalized_shape, dtype=torch.float32, device=device))

    def forward(self, x):
        return self.scale * self.norm(x)
    

class DualHeadSimpleAttention(nn.Module):
    # input shape: (batch_size, in_features, seq_len)
    def __init__(self, in_features, embed_features, out_features, device):
        super().__init__()
        self.device = device
        self.embed_features = embed_features
        self.out_features = out_features
        self.conv1d_q = nn.Conv1d(in_channels=in_features, out_channels=embed_features * 2, kernel_size=1, bias=False)
        self.conv1d_k = nn.Conv1d(in_channels=in_features, out_channels=embed_features * 2, kernel_size=1, bias=False)
        self.conv1d_v = nn.Conv1d(in_channels=in_features, out_channels=out_features * 2, kernel_size=1, bias=False)

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[2]
            
        q = self.conv1d_q(x)  # (batch_size, 2 * embed_features, seq_len)
        k = self.conv1d_k(x)  # (batch_size, 2 * embed_features, seq_len)
        v = self.conv1d_v(x)  # (batch_size, 2 * out_features, seq_len)
        
        q = q.reshape(batch_size, seq_len, 2, -1).transpose(1, 2)  # (batch_size, 2, seq_len, embed_features)
        k = k.reshape(batch_size, seq_len, 2, -1).transpose(1, 2)  # (batch_size, 2, seq_len, embed_features)
        v = v.reshape(batch_size, seq_len, 2, -1).transpose(1, 2)  # (batch_size, 2, seq_len, out_features)

        scale = torch.sqrt(torch.tensor(q.size(-1), dtype=torch.float32, device=self.device))
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        
        results = torch.matmul(attn, v)  # (batch_size, 2, seq_len, out_features)
        
        results = results.reshape(batch_size, -1, seq_len)  # (batch_size, 2 * out_features, seq_len)
        results = F.adaptive_avg_pool1d(results, 1).squeeze(-1)  # (batch_size, 2 * out_features)
        
        return results


class DLModel(nn.Module):
    def __init__(self, device, mode):
        super(DLModel, self).__init__()
        self.mode = mode
        self.device = device
        self.act = MyActivationLayer(device)
        self.temp_params = [
            self.act.temp_mu,
            self.act.temp_sigma,
        ]
        self.precip_params = [
            self.act.precip_mu,
            self.act.precip_sigma,
        ]
        
        self.circular_pad = nn.CircularPad2d(padding=(2, 0, 0, 0))

        # act left branch
        self.grouped_conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, groups=2, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(num_features=32)
        # todo: relu

        # act right branch
        self.grouped_conv3 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(3, 5), stride=(1, 3), dilation=(3, 1), groups=2, bias=False)
        # todo: reshape

        # relu1 left branch
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=(1, 3), dilation=(3, 1), bias=False)
        self.batch_norm2 = nn.BatchNorm1d(num_features=32)
        # todo: relu

        # relu1 mid branch
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=(1, 3), dilation=(2, 1), bias=False)
        # todo: reshape
        self.batch_norm3 = nn.BatchNorm1d(num_features=96)
        # todo: relu

        # relu1 right branch
        self.grouped_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 3), stride=(1, 3), groups=32, bias=False)
        # todo: reshape

        # todo: concatenate
        self.layer_norm = nn.GroupNorm(num_groups=1, num_channels=230, affine=False)

        # layer norm left branch (feature weights)
        self.attention = DualHeadSimpleAttention(in_features=230, embed_features=40, out_features=60, device=device)
        self.group_norm = nn.GroupNorm(num_groups=2, num_channels=120)
        # todo: relu
        self.bn_score = nn.BatchNorm1d(num_features=120, affine=False)

        # layer norm right branch (features)
        self.conv1d = nn.Conv1d(in_channels=230, out_channels=60, kernel_size=1, bias=False)
        # todo: pool and concat
        self.batch_norm4 = BatchNormWithScale(normalized_shape=120, device=device)
        # todo: softsign
        if self.mode == 'train':
            self.linear = orthogonal(nn.Linear(in_features=120, out_features=60, bias=False))
        else:
            self.linear = nn.Linear(in_features=120, out_features=60, bias=False)

        # todo: multiply
        if self.mode == 'train':
            self.linear1 = weight_norm(nn.Linear(in_features=120, out_features=240), dim=1)
        else:
            self.linear1 = nn.Linear(in_features=120, out_features=240)

        # todo: relu
        # todo: transpose
        self.dropout = nn.Dropout1d(p=0.5)
        # todo: transpose
        self.linear2 = nn.Linear(in_features=240, out_features=14)
        # todo: softmax

        self.cluster = StructuredKMeans(None, 26, 60, device=device)
        self.cluster_params = [
            self.cluster.centers,
        ]
        # self.cluster.centers.requires_grad = False

        exclude_ids = {id(param) for param in self.temp_params + self.precip_params + self.cluster_params}
        self.other_params = [p for p in self.parameters() if id(p) not in exclude_ids]

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.act(x)
        x = self.circular_pad(x)

        # act left branch
        x1 = self.grouped_conv1(x)
        x1 = self.batch_norm1(x1)
        x1 = F.relu_(x1)

        # act right branch
        x2 = self.grouped_conv3(x)
        x2 = x2.reshape(batch_size, -1, 4)

        # relu1 left branch
        x1_1 = self.conv1(x1)
        x1_1 = x1_1.squeeze(2)
        x1_1 = self.batch_norm2(x1_1)
        x1_1 = F.relu_(x1_1)

        # relu1 mid branch
        x1_2 = self.conv2(x1)
        x1_2 = x1_2.reshape(batch_size, -1, 4)
        x1_2 = self.batch_norm3(x1_2)
        x1_2 = F.relu_(x1_2)

        # relu1 right branch
        x1_3 = self.grouped_conv2(x1)
        x1_3 = x1_3.reshape(batch_size, -1, 4)

        # concatenate
        x = torch.cat([x1_1, x1_2, x1_3, x2], dim=1)
        x = self.layer_norm(x)
        # layer norm right branch (features)
        x_f = self.conv1d(x)
        x_f_max = torch.max(x_f, dim=2).values
        x_f_avg = torch.mean(x_f, dim=2)
        x_f = torch.cat([x_f_max, x_f_avg], dim=1)
        x_f = self.batch_norm4(x_f)
        x_f = F.softsign(x_f)

        if self.mode == 'inference':   
            x_f = self.linear(x_f * self.bn_score.running_mean)
            _, prob = self.cluster(x_f)
            # veg = torch.argmax(x, dim=1)
            return -x_f[:, 0].numpy(force=True), -x_f[:, 1].numpy(force=True), prob.numpy(force=True)    #, veg.numpy(force=True)
        else:
            # layer norm left branch (feature weights)
            x_w = self.attention(x)
            x_w = self.group_norm(x_w)
            x_w = F.relu_(x_w)
            _ = self.bn_score(x_w)

            # multiply
            x = x_w * x_f
            x = self.linear1(x)
            x = F.relu_(x)

            x = x.transpose(0, 1)
            x = self.dropout(x)
            x = x.transpose(0, 1)
            x = self.linear2(x)
            x_f = self.linear(x_f * x_w.mean(dim=0))
            cluster_loss, sample_loss, centroids_mean_distance, centroids_std_distance = self.cluster.compute_loss(x_f)
            veg_prob = F.softmax(x, dim=1)
            return x_f, cluster_loss, sample_loss, centroids_mean_distance, centroids_std_distance, veg_prob

    def init_weights(self, init_centroids, k, T):
        nn.init.xavier_uniform_(self.grouped_conv1.weight, gain=1.0)
        nn.init.xavier_uniform_(self.grouped_conv2.weight, gain=1.0)
        nn.init.xavier_uniform_(self.grouped_conv3.weight, gain=1.0)
        nn.init.xavier_uniform_(self.conv1.weight, gain=1.0)
        nn.init.xavier_uniform_(self.conv2.weight, gain=1.0)
        nn.init.xavier_uniform_(self.conv1d.weight, gain=1.0)
        nn.init.xavier_uniform_(self.attention.conv1d_q.weight, gain=1.0)
        nn.init.xavier_uniform_(self.attention.conv1d_k.weight, gain=1.0)
        nn.init.xavier_uniform_(self.attention.conv1d_v.weight, gain=1.0)
        nn.init.orthogonal_(self.linear.weight, gain=1.0)
        nn.init.xavier_uniform_(self.linear1.weight, gain=1.0)
        nn.init.xavier_uniform_(self.linear2.weight, gain=1.0)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)

        self.cluster.centers = torch.nn.Parameter(torch.from_numpy(init_centroids).to(self.device), requires_grad=True)
        self.cluster.k = k
        self.cluster.T = T


if __name__ == "__main__":
    import torch.optim as optim
    from scipy.io import loadmat, savemat
    import numpy as np
    np.set_printoptions(precision=4, suppress=True)
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm

    torch.set_float32_matmul_precision('high')

    # code for migrating and fine-tuning MATLAB-pretrained model
    batch_size = 256
    init_lr = 1e-3
    weight_decay = 0.0
    epochs = 30
    # lr_step = 10
    # lr_gamma = 1/3
    cluster_loss_weight = 0.1
    feature_loss_weight = 10.0
    k = 0.1
    T = 0.21
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.cuda.empty_cache()
    
    print("Using device:", device)

    # Sorry I can't provide the dataset in the GitHub repo
    # You can download it from the links
    # Climate data: CRU TS v4.06
    # https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.06/
    # Land Cover data: MCD12C1
    # https://www.earthdata.nasa.gov/data/catalog/lpcloud-mcd12c1-061
    # The features data is obtained by passing the inputs to the MATLAB pretrained model
    data = loadmat("data.mat")    
    inputs = data["inputs"]       # model inputs (batch, 3, 12) - climate normals
    features = data["features"]   # MATLAB-pretrained climate features (batch, 60), teacher knowledge
    init_centroids = data["new_centroid"]  # MATLAB-pretrained cluster centroids (26, 60)
    targets = data["targets"]    # model targets (batch, 14) - land cover classes proportions

    print(inputs.shape, features.shape, init_centroids.shape, targets.shape)
    del data

    model = DLModel(device, 'train')
    model.init_weights(init_centroids, k, T)
    model = model.to(device)
    model = torch.compile(model)

    inputs = torch.from_numpy(inputs).float().pin_memory().to(device, non_blocking=True)
    features = torch.from_numpy(features).float().pin_memory().to(device, non_blocking=True)
    targets = torch.from_numpy(targets).float().pin_memory().to(device, non_blocking=True)
    # Last year in the dataset
    valid_inputs = inputs[-67213:, :, :]
    valid_features = features[-67213:, :]
    valid_targets = targets[-67213:, :]

    ds = TensorDataset(inputs, features, targets)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    
    bceloss = nn.BCELoss()
    mseloss = nn.MSELoss()
    optimizer = optim.Adam([{'params': model.temp_params, 'lr': init_lr * 30},
                           {'params': model.precip_params, 'lr': init_lr * 200},
                           {'params': model.cluster.centers, 'lr': init_lr / 5},
                           {'params': model.other_params}],
                           lr=init_lr,
                           weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=init_lr * 0.01)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, lr_step, lr_gamma)

    # model.mode = 'train'
    model.eval()
    with torch.no_grad():
        output_features, cluster_loss, sample_loss, centroids_mean_distance, centroids_std_distance, veg_prob = model(valid_inputs)
        bce_loss = bceloss(veg_prob, valid_targets) - bceloss(valid_targets, valid_targets)
        mse_loss = mseloss(output_features, valid_features)
        loss = bce_loss + cluster_loss_weight * cluster_loss + feature_loss_weight * mse_loss
        print("Before Training")
        print(f'Total Loss: {loss.item()}, Veg Loss: {bce_loss.item()}, Feature Loss: {mse_loss.item()}')
        print(f'Sample Loss: {sample_loss.item()}, Centroids Mean Distance: {centroids_mean_distance.item()}, Centroids Std Distance: {centroids_std_distance.item()}')

    best_loss = loss.item()
    best_model = model.state_dict()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} / {epochs}")

        model.train()
        if epoch == 15:
            print("-----------------------")

        for input, feature, target in tqdm(dl, desc="Training"):
            optimizer.zero_grad()
            output_features, cluster_loss, _, _, _, veg_prob = model(input)
            bce_loss = bceloss(veg_prob, target) - bceloss(target, target)
            mse_loss = mseloss(output_features, feature)
            loss = bce_loss + cluster_loss_weight * cluster_loss + feature_loss_weight * mse_loss
            loss.backward()

            if epoch < 15:
                model.cluster.centers.grad = None

            optimizer.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            output_features, cluster_loss, sample_loss, centroids_mean_distance, centroids_std_distance, veg_prob = model(valid_inputs)
            bce_loss = bceloss(veg_prob, valid_targets) - bceloss(valid_targets, valid_targets)
            mse_loss = mseloss(output_features, valid_features)
            loss = bce_loss + cluster_loss_weight * cluster_loss + feature_loss_weight * mse_loss
            print(f'Total Loss: {loss.item()}, Veg Loss: {bce_loss.item()}, Feature Loss: {mse_loss.item()}')
            print(f'Sample Loss: {sample_loss.item()}, Centroids Mean Distance: {centroids_mean_distance.item()}, Centroids Std Distance: {centroids_std_distance.item()}')
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model = model.state_dict()

    best_model.linear = remove_parametrizations(best_model.linear, 'weight')
    best_model.linear1 = remove_parametrizations(best_model.linear1, 'weight')
    torch.save(best_model, "best_model.pth")
        
    model.mode = 'inference'
    model.eval()
    test_data = loadmat("test_data.mat")
    test_data = test_data["test_input"]
    test_data = torch.from_numpy(test_data).float().pin_memory().to(device, non_blocking=True)
    _, _, res = model(test_data)
    centroids = model.cluster.centers.numpy(force=True)
    savemat("result.mat", {"prob": res, "centroid": centroids})