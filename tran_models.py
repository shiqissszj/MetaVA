import torch
from torch import nn

from torch.utils.data import Dataset



class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return (torch.tensor(self.data[index], dtype=torch.float), torch.tensor(self.label[index], dtype=torch.long))

    def __len__(self):
        return len(self.data)


class ecgTransForm(nn.Module):
    def __init__(self, input_channels, mid_channels, trans_dim, num_heads, dropout, num_classes, stride):
        super(ecgTransForm, self).__init__()

        # Multi-scale convolution layers
        filter_sizes = [5, 9, 11]
        self.conv1 = nn.Conv1d(input_channels, mid_channels,
                               kernel_size=filter_sizes[0], stride=stride,
                               padding=filter_sizes[0] // 2, bias=False)
        self.conv2 = nn.Conv1d(input_channels, mid_channels,
                               kernel_size=filter_sizes[1], stride=stride,
                               padding=filter_sizes[1] // 2, bias=False)
        # self.conv3 = nn.Conv1d(input_channels, mid_channels,
        #                        kernel_size=filter_sizes[2], stride=stride,
        #                        padding=filter_sizes[2] // 2, bias=False)

        # Shared processing layers
        self.bn = nn.BatchNorm1d(mid_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.do = nn.Dropout(dropout)

        # Secondary conv block (adjust channels to trans_dim)
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(mid_channels, trans_dim, kernel_size=8, padding=4, bias=False),
            nn.BatchNorm1d(trans_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(50)  # unify sequence length to 50
        )

        # Transformer encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=trans_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        # Classification head
        self.clf = nn.Linear(trans_dim, num_classes)

    def forward(self, x_in):
        # Multi-scale feature extraction
        x1 = self.conv1(x_in)
        x2 = self.conv2(x_in)
        # x3 = self.conv3(x_in)

        # Average fusion + feature processing
        x_concat = torch.mean(torch.stack([x1, x2], dim=2), dim=2)
        x = self.do(self.mp(self.relu(self.bn(x_concat))))

        # Adjust channel dimension
        x = self.conv_block2(x)  # [batch, trans_dim, 50]

        # Transformer processing
        x = x.permute(0, 2, 1)  # [batch, 50, trans_dim]
        x = self.transformer_encoder(x)

        # Classification output
        x = x.mean(dim=1)  # global average pooling
        return self.clf(x)

if __name__ == '__main__':
    from torchsummary import summary

    model = ecgTransForm(input_channels=1, mid_channels=32, trans_dim=64, num_heads=8, dropout=0.1, num_classes=2,
                         stride=1)
    # summary(model, input_size=(32, 400))  # example input shape
    print(model)
