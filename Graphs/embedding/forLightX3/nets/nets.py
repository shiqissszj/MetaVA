import torch
import torch.nn as nn
from .backbones import LightSEResNet18


class LightX3ECG(nn.Module):
    def __init__(self, base_channels=64, num_classes=2, num_leads=2):
        super(LightX3ECG, self).__init__()
        self.num_leads = num_leads

        if num_leads == 1:
            self.backbone_0 = LightSEResNet18(base_channels)
        elif num_leads == 2:
            self.backbone_0 = LightSEResNet18(base_channels)
            self.backbone_1 = LightSEResNet18(base_channels)

        feature_dim = base_channels * num_leads * 8  # 单导联: 512, 双导联: 1024
        self.lw_attention = nn.Sequential(
            nn.Linear(feature_dim, base_channels * 8),
            nn.BatchNorm1d(base_channels * 8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(base_channels * 8, feature_dim),  # 输出与 features 匹配
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(base_channels * 8, num_classes),  # 输入固定为 base_channels * 8
        )

    def forward(self, input, return_attention_scores=False):
        if self.num_leads == 1:
            features_0 = self.backbone_0(input[:, 0, :].unsqueeze(1)).squeeze(2)  # (batch_size, base_channels * 8)
            features = features_0
        elif self.num_leads == 2:
            features_0 = self.backbone_0(input[:, 0, :].unsqueeze(1)).squeeze(2)  # (batch_size, base_channels * 8)
            features_1 = self.backbone_1(input[:, 1, :].unsqueeze(1)).squeeze(2)  # (batch_size, base_channels * 8)
            features = torch.cat([features_0, features_1], dim=1)  # (batch_size, base_channels * 2 * 8)

        attention_scores = torch.sigmoid(self.lw_attention(features))  # (batch_size, feature_dim)
        merged_features = (features * attention_scores)  # (batch_size, feature_dim)

        # 调整 merged_features 为 (batch_size, base_channels * 8)
        if self.num_leads == 2:
            # 双导联时，feature_dim = base_channels * 2 * 8，需压缩到 base_channels * 8
            merged_features = merged_features.view(-1, 2, 64 * 8).sum(
                dim=1)  # (batch_size, base_channels * 8)
        # 单导联时，feature_dim 已等于 base_channels * 8，无需调整

        output = self.classifier(merged_features)  # (batch_size, num_classes)

        if not return_attention_scores:
            return output
        else:
            return output, attention_scores

    # def get_embedding(self, input):
    #     if not hasattr(self, 'num_leads'):
    #         self.num_leads = 1
    #     if not hasattr(self, 'backbone_0'):
    #         self.backbone_0 = LightSEResNet18(base_channels=64).to('cuda')
    #     if not hasattr(self, 'backbone_1'):
    #         self.backbone_1 = LightSEResNet18(base_channels=64).to('cuda')
    #     if self.num_leads == 1:
    #         features_0 = self.backbone_0(input[:, 0, :].unsqueeze(1)).squeeze(2)
    #         features = features_0
    #     elif self.num_leads == 2:
    #         features_0 = self.backbone_0(input[:, 0, :].unsqueeze(1)).squeeze(2)
    #         features_1 = self.backbone_1(input[:, 1, :].unsqueeze(1)).squeeze(2)
    #         features = torch.cat([features_0, features_1], dim=1)
    #
    #     attention_scores = torch.sigmoid(self.lw_attention(features))
    #     merged_features = features * attention_scores
    #
    #     if self.num_leads == 2:
    #         merged_features = merged_features.view(-1, 2, 64 * 8).sum(dim=1)
    #     return merged_features  # 返回嵌入特征

    def get_embedding(self, input):
        # 如果 num_leads 未定义，默认为 1
        if not hasattr(self, 'num_leads'):
            self.num_leads = 1
        # 如果 backbone_0 不存在，动态创建
        if not hasattr(self, 'backbone_0'):
            self.backbone_0 = LightSEResNet18(base_channels=64).to(input.device)
        # 如果 num_leads == 2，确保 backbone_1 也存在
        if self.num_leads == 2 and not hasattr(self, 'backbone_1'):
            self.backbone_1 = LightSEResNet18(base_channels=64).to(input.device)

        # 原有逻辑
        features_0 = self.backbone_0(input[:, 0, :].unsqueeze(1)).squeeze(2)
        if self.num_leads == 2:
            features_1 = self.backbone_1(input[:, 1, :].unsqueeze(1)).squeeze(2)
            features = torch.cat([features_0, features_1], dim=-1)
        else:
            features = features_0
        return features