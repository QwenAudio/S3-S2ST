import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_heads=4, num_layers=1):
        """
        小型Transformer模型用于片段特征提取
        
        参数:
            input_dim: 输入特征维度 (D)
            hidden_dim: Transformer隐藏层维度
            num_heads: 注意力头数
            num_layers: Transformer层数
        """
        super().__init__()
        
        # 输入投影层
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出投影层
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # 可学习的[CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
    def forward(self, x, mask=None):
        """
        前向传播
        
        参数:
            x: 输入张量 [B, L, D]
            mask: 填充掩码 [B, L]
        
        返回:
            聚合特征 [B, D]
        """
        # 输入投影
        x = self.input_proj(x)  # [B, L, hidden_dim]
        
        # 添加[CLS] token
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, L+1, hidden_dim]
        
        # 更新掩码（为CLS token添加False）
        if mask is not None:
            cls_mask = torch.zeros(x.size(0), 1, dtype=torch.bool, device=x.device)
            mask = torch.cat([cls_mask, mask], dim=1)

        # 通过Transformer
        x = self.transformer(x, src_key_padding_mask=mask)

        # 取CLS token的输出
        cls_output = x[:, 0, :]  # [B, hidden_dim]

        # 输出投影
        return self.output_proj(cls_output)  # [B, D]
