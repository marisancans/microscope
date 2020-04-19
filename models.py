from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import torch
from helpers import unflatten_sequences

def encoder_block(in_channels, features, name):
    return nn.Sequential(
        OrderedDict(
            [
                (
                    name + "conv1",
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=features,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                ),
                (name + "norm1", nn.BatchNorm2d(num_features=features)),
                (name + "relu1", nn.ReLU(inplace=True)),
                (
                    name + "conv2",
                    nn.Conv2d(
                        in_channels=features,
                        out_channels=features,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                ),
                (name + "norm2", nn.BatchNorm2d(num_features=features)),
                (name + "relu2", nn.ReLU(inplace=True)),
            ]
        )
    )


class TileSniperModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=32):
        super(TileSniperModel, self).__init__()

        # encoder
        self.encoder1 = encoder_block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = encoder_block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc_score = nn.Linear(in_features=features * 2, out_features=1)
        self.fc_coords = nn.Linear(in_features=features * 2, out_features=2)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc1 = F.relu(enc1)

        enc2 = self.pool1(enc1)
        enc2 = self.encoder2(enc2)
        enc2 = F.relu(enc2)

        avg_pooled = self.avg_pool(enc2)
        avg_pooled = avg_pooled.squeeze(-1).squeeze(-1)

        score = self.fc_score(avg_pooled)
        coords = self.fc_coords(avg_pooled)

        score = F.tanh(score)
        coords = F.sigmoid(coords)

        return score, coords


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, device='cpu', num_layers=1):
        super(LSTMEncoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False)
        self.device = device

    def forward(self, x, batch_size):
        # Initialize cell state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)

        all_timesteps, hidden_last_timestep = self.lstm(x, (h0, c0))

        return all_timesteps


class DirectionModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=32, device='cpu'):
        super(DirectionModel, self).__init__()

        # encoder
        self.encoder1 = encoder_block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = encoder_block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = encoder_block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = encoder_block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.lstm = LSTMEncoder(input_size=features * 8, hidden_size=features * 8, device=device)

        self.fc_score = nn.Linear(in_features=features * 8, out_features=1)
        self.fc_coords = nn.Linear(in_features=features * 8, out_features=2)

    def forward(self, flattened_patches, flat_ids):
        enc1 = self.encoder1(flattened_patches)
        enc1 = F.relu(enc1)

        enc2 = self.pool1(enc1)
        enc2 = self.encoder2(enc2)
        enc2 = F.relu(enc2)

        enc3 = self.pool2(enc2)
        enc3 = self.encoder3(enc3)
        enc3 = F.relu(enc3)

        enc4 = self.pool3(enc3)
        enc4 = self.encoder4(enc4)
        enc4 = F.relu(enc4)

        avg_pooled = self.avg_pool(enc4)
        avg_pooled = avg_pooled.squeeze(-1).squeeze(-1) # remove H and W

        sequences = unflatten_sequences(avg_pooled, flat_ids) 

        padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=False)
        lengths = [x.size(0) for x in sequences]
        pack = nn.utils.rnn.pack_padded_sequence(padded_sequences, lengths, batch_first=False, enforce_sorted=False) # enforce_sorted only necessary for ONNX export

        lstm_out = self.lstm.forward(pack, batch_size=len(sequences))
        unpacked, unpacked_len = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=False) # Inverse operation to 'pack_padded_sequence'
        
        # (Seq_len, B, features) --> (B, Seq_len, features)
        unpacked = unpacked.permute(1, 0, 2)

        unflatted_lstm = [seq[:idx] for seq, idx in zip(unpacked, unpacked_len)]
        merged_timesteps = [torch.mean(x, dim=0) for x in unflatted_lstm]
        merged_timesteps = torch.stack(merged_timesteps)

        scores = self.fc_score(merged_timesteps)
        coords = self.fc_coords(merged_timesteps)

        scores = F.tanh(scores)
        coords = F.sigmoid(coords)

        return scores, coords




# class GlueModel(nn.Module):
#     def __init__(self, in_channels=1, out_channels=1, features=32, device='cpu'):
#         super(GlueModel, self).__init__()
#             self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=1, padding=1, bias=False)
#             self.norm1 = nn.BatchNorm2d(num_features=features)
#             self.relu1 = nn.ReLU(inplace=True)
#             self.conv2 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False)
#             self.norm2 = nn.BatchNorm2d(num_features=features)
#             self.relu2 = nn.ReLU(inplace=True)            


#     def forward(self, flattened_patches, flat_ids):
#         enc1 = self.encoder1(flattened_patches)
#         enc1 = self.bn_enc1(enc1)
#         enc1 = F.relu(enc1)

#         enc2 = self.pool1(enc1)
#         enc2 = self.encoder2(enc2)
#         enc2 = self.bn_enc2(enc2)
#         enc2 = F.relu(enc2)

#         enc3 = self.pool2(enc2)
#         enc3 = self.encoder3(enc3)
#         enc3 = self.bn_enc3(enc3)
#         enc3 = F.relu(enc3)

#         enc4 = self.pool3(enc3)
#         enc4 = self.encoder4(enc4)
#         enc4 = self.bn_enc4(enc4)
#         enc4 = F.relu(enc4)

#         avg_pooled = self.avg_pool(enc4)
#         avg_pooled = avg_pooled.squeeze(-1).squeeze(-1) # remove H and W

#         sequences = unflatten_sequences(avg_pooled, flat_ids) 

#         padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=False)
#         lengths = [x.size(0) for x in sequences]
#         pack = nn.utils.rnn.pack_padded_sequence(padded_sequences, lengths, batch_first=False, enforce_sorted=False) # enforce_sorted only necessary for ONNX export

#         lstm_out = self.lstm.forward(pack, batch_size=len(sequences))
#         unpacked, unpacked_len = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=False) # Inverse operation to 'pack_padded_sequence'
        
#         # (Seq_len, B, features) --> (B, Seq_len, features)
#         unpacked = unpacked.permute(1, 0, 2)

#         unflatted_lstm = [seq[:idx] for seq, idx in zip(unpacked, unpacked_len)]
#         merged_timesteps = [torch.mean(x, dim=0) for x in unflatted_lstm]
#         merged_timesteps = torch.stack(merged_timesteps)

#         scores = self.fc_score(merged_timesteps)
#         angles = self.fc_angle(merged_timesteps)

#         scores = F.tanh(scores)
#         angles = F.sigmoid(angles)

#         return scores, angles

