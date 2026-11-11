import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .utils.mask import make_pad_mask

class PositionalEncoding(nn.Module):
    """
    固定位置编码，用于为序列添加位置信息。
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0)]
        return x

class TransformerVAE(nn.Module):
    """
    基于Transformer的VAE模型，编码器和解码器均为Transformer结构。
    """
    def __init__(self, input_dim, latent_dim, d_model_encoder=128, d_model_decoder=128, 
                 encoder_layers=2, decoder_layers=2, nhead=8, dim_feedforward=256, dropout=0.1):
        super(TransformerVAE, self).__init__()
        
        # 编码器部分
        self.enc_in_proj = nn.Linear(input_dim, d_model_encoder)
        self.encoder_pos = PositionalEncoding(d_model_encoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model_encoder, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, encoder_layers)
        self.fc_mu = nn.Linear(d_model_encoder, latent_dim)
        self.fc_sigma = nn.Linear(d_model_encoder, latent_dim)
        
        # 解码器部分
        self.dec_in_proj = nn.Linear(latent_dim, d_model_decoder)
        self.decoder_pos = PositionalEncoding(d_model_decoder)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model_decoder, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=False
        )
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, decoder_layers)
        self.dec_out_proj = nn.Linear(d_model_decoder, input_dim)
    
    def encode(self, x):
        """
        将输入序列编码为潜在空间的均值和方差。
        输入 x: [N, T, input_dim]
        输出 mu, sigma: [N, T, latent_dim]
        """
        x = self.enc_in_proj(x)  # [N, T, d_model_encoder]
        x = x.permute(1, 0, 2)  # [T, N, d_model_encoder]
        x = self.encoder_pos(x)
        encoded = self.transformer_encoder(x)  # [T, N, d_model_encoder]
        mu = self.fc_mu(encoded)  # [T, N, latent_dim]
        sigma = self.fc_sigma(encoded)  # [T, N, latent_dim]
        mu = mu.permute(1, 0, 2)  # [N, T, latent_dim]
        sigma = sigma.permute(1, 0, 2)
        return mu, sigma
    
    def reparameterize(self, mu, sigma):
        """
        重参数化技巧，从分布中采样潜在变量。
        """
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """
        将潜在变量解码为重建的输入序列。
        输入 z: [N, T, latent_dim]
        输出 recon_x: [N, T, input_dim]
        """
        z = self.dec_in_proj(z)  # [N, T, d_model_decoder]
        z = z.permute(1, 0, 2)  # [T, N, d_model_decoder]
        z = self.decoder_pos(z)
        decoded = self.transformer_decoder(z)  # [T, N, d_model_decoder]
        output = self.dec_out_proj(decoded)  # [T, N, input_dim]
        output = output.permute(1, 0, 2)  # [N, T, input_dim]
        return output
    
    def forward(self, x):
        mu, sigma = self.encode(x)
        z = self.reparameterize(mu, sigma)
        recon_x = self.decode(z)
        return recon_x, mu, sigma

def vae_loss_function(recon_x, x, mu, sigma, x_len):
    """
    VAE的损失函数，包括重构损失和KL散度。
    """
    # 重构损失（均方误差）
    mask = (~make_pad_mask(x_len.squeeze(-1))).to(x)
    # recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    recon_loss = (F.mse_loss(recon_x, x, reduction="none") * mask.unsqueeze(2)).sum() / (torch.sum(mask) * x.shape[2])
    # KL散度（sigma为log variance）
    #KLD = -0.5 * torch.mean(1 + sigma - mu.pow(2) - sigma.exp())
    KLD = ((-0.5 * (1 + sigma - mu.pow(2) - sigma.exp())) * mask.unsqueeze(2)).sum() / (torch.sum(mask) * mu.shape[2])
    return recon_loss, KLD
