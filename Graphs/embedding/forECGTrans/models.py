import torch
from torch import nn


class ecgTransForm(nn.Module):
    def __init__(self, configs, hparams):
        super(ecgTransForm, self).__init__()

        filter_sizes = [5, 9, 11]
        self.conv1 = nn.Conv1d(configs["input_channels"], configs["mid_channels"], kernel_size=filter_sizes[0],
                               stride=configs["stride"], bias=False, padding=(filter_sizes[0] // 2))
        self.conv2 = nn.Conv1d(configs["input_channels"], configs["mid_channels"], kernel_size=filter_sizes[1],
                               stride=configs["stride"], bias=False, padding=(filter_sizes[1] // 2))
        self.conv3 = nn.Conv1d(configs["input_channels"], configs["mid_channels"], kernel_size=filter_sizes[2],
                               stride=configs["stride"], bias=False, padding=(filter_sizes[2] // 2))

        self.bn = nn.BatchNorm1d(configs["mid_channels"])
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.do = nn.Dropout(configs["dropout"])

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs["mid_channels"], configs["mid_channels"] * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs["mid_channels"] * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs["mid_channels"] * 2, configs["final_out_channels"], kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs["final_out_channels"]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.inplanes = 128
        self.crm = self._make_layer(SEBasicBlock, 128, 3)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=configs["trans_dim"], nhead=configs["num_heads"], batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

        self.aap = nn.AdaptiveAvgPool1d(1)
        # 修改为二分类输出：输出维度为 1
        self.clf = nn.Linear(hparams["feature_dim"], 1)

    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x_in):
        # 初始处理（例如 CRM）
        x1 = self.conv1(x_in)  # (32, 32, 400)
        x2 = self.conv2(x_in)
        x3 = self.conv3(x_in)
        # print(f"Conv1 shape: {x1.shape}")

        x_concat = torch.mean(torch.stack([x1, x2, x3], dim=2), dim=2)  # (32, 32, 400)
        x_concat = self.do(self.mp(self.relu(self.bn(x_concat))))  # (32, 32, 201)
        # print(f"After concat and pool: {x_concat.shape}")

        x = self.conv_block2(x_concat)  # (32, 64, 102)
        # print(f"After conv_block2: {x.shape}")
        x = self.conv_block3(x)  # (32, 128, 52)
        # print(f"After conv_block3: {x.shape}")
        x = self.crm(x)  # (32, 128, 52)
        # print(f"After CRM: {x.shape}")

        # 为 Transformer 准备：(batch_size, sequence_length, d_model)
        x = x.permute(0, 2, 1)  # (32, 52, 128)

        # Transformer 编码
        x1 = self.transformer_encoder(x)  # (32, 52, 128)
        x2 = self.transformer_encoder(torch.flip(x, [1]))  # (32, 52, 128)
        x = x1 + x2  # (32, 52, 128)

        # 为池化调整维度：(batch_size, d_model, sequence_length)
        x = x.permute(0, 2, 1)  # (32, 128, 52)

        # 在序列长度上进行自适应平均池化
        x = self.aap(x)  # (32, 128, 52) -> (32, 128, 1)

        # 展平
        x_flat = x.reshape(x.shape[0], -1)  # (32, 128)

        # 分类器
        x_out = self.clf(x_flat)  # (32, 128) -> (32, 1)
        return x_out

    def get_embedding(self, x_in):
        # 初始卷积处理
        x1 = self.conv1(x_in)  # (batch_size, 32, 400)
        x2 = self.conv2(x_in)  # (batch_size, 32, 400)
        x3 = self.conv3(x_in)  # (batch_size, 32, 400)
        x_concat = torch.mean(torch.stack([x1, x2, x3], dim=2), dim=2)  # (batch_size, 32, 400)
        x_concat = self.do(self.mp(self.relu(self.bn(x_concat))))  # (batch_size, 32, 201)

        # 卷积块
        x = self.conv_block2(x_concat)  # (batch_size, 64, 102)
        x = self.conv_block3(x)  # (batch_size, 128, 52)
        x = self.crm(x)  # (batch_size, 128, 52)

        # 为 Transformer 准备：调整维度为 (batch_size, sequence_length, d_model)
        x = x.permute(0, 2, 1)  # (batch_size, 52, 128)

        # Transformer 编码
        x1 = self.transformer_encoder(x)  # (batch_size, 52, 128)
        x2 = self.transformer_encoder(torch.flip(x, [1]))  # (batch_size, 52, 128)
        x = x1 + x2  # (batch_size, 52, 128)

        # 为池化调整维度：(batch_size, d_model, sequence_length)
        x = x.permute(0, 2, 1)  # (batch_size, 128, 52)

        # 自适应平均池化
        x = self.aap(x)  # (batch_size, 128, 1)

        # 展平为嵌入特征
        x_flat = x.reshape(x.shape[0], -1)  # (batch_size, 128)
        return x_flat  # 返回嵌入特征


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=4):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out