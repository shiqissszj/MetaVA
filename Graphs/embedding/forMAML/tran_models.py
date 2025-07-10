import torch
from torch import nn


class ecgTransForm(nn.Module):
    def __init__(self, input_channels, mid_channels, trans_dim, num_heads, dropout, num_classes, stride):
        super(ecgTransForm, self).__init__()

        # 多尺度卷积层
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

        # 公共处理层
        self.bn = nn.BatchNorm1d(mid_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.do = nn.Dropout(dropout)

        # 次级卷积块（调整输出通道为trans_dim）
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(mid_channels, trans_dim, kernel_size=8, padding=4, bias=False),
            nn.BatchNorm1d(trans_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(50)  # 统一序列长度为50
        )

        # Transformer编码器
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=trans_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        # 分类层
        self.clf = nn.Linear(trans_dim, num_classes)

    def forward(self, x_in):
        # 多尺度特征提取
        x1 = self.conv1(x_in)
        x2 = self.conv2(x_in)
        # x3 = self.conv3(x_in)

        # 平均融合+特征处理
        x_concat = torch.mean(torch.stack([x1, x2], dim=2), dim=2)
        x = self.do(self.mp(self.relu(self.bn(x_concat))))

        # 调整通道维度
        x = self.conv_block2(x)  # [batch, trans_dim, 50]

        # Transformer处理
        x = x.permute(0, 2, 1)  # [batch, 50, trans_dim]
        x = self.transformer_encoder(x)

        # 分类输出
        x = x.mean(dim=1)  # 全局平均池化
        return self.clf(x)

    def get_embedding(self, x):
        """
        提取中间嵌入，用于 UMAP 可视化。
        输入 x: [batch_size, input_channels, sequence_length]
        输出: [batch_size, embedding_dim]
        """
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x_concat = torch.mean(torch.stack([x1, x2], dim=2), dim=2)
        x = self.do(self.mp(self.relu(self.bn(x_concat))))
        x = self.conv_block2(x)
        x = x.permute(0, 2, 1)  # [batch_size, sequence_length, channels]
        x = self.transformer_encoder(x)
        embedding = x.mean(dim=1)  # 全局平均池化，输出 [batch_size, embedding_dim]
        return embedding