import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Define the autoencoder architecture
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AutoEncoder, self).__init__()
        # self.encoder = Dense(hidden_dim, activation='relu')
        # self.decoder = Dense(input_dim, activation='relu')
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )



    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# input_dim = s_dim - 1 + a_dim
# hidden_dim = 128
# output_dim = s_dim - 1 + 1
# auto_model = AutoEncoder(input_dim, hidden_dim, output_dim)
# bs_ba = np.concatenate((bs[:, 1:], ba), axis=1) # 去除time的一列
# br_bs_ = np.concatenate((br, bs_[:, 1:]), axis=1)
# # Convert the dataset to PyTorch tensors
# bs_ba_tensor = torch.tensor(bs_ba, dtype=torch.float32)
# br_bs_tensor = torch.tensor(br_bs_, dtype=torch.float32)
# dataset = TensorDataset(bs_ba_tensor, br_bs_tensor)
# # Train the model
# epochs = 10
# batch_size = 64
# learning_rate = 0.001
# train_autoencoder(auto_model, dataset, epochs, batch_size, learning_rate)