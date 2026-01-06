from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MixerLayer, TimeBatchNorm2d, feature_to_time, time_to_feature



class CTFT_TSMixer_GR(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        prediction_length: int,
        input_channels: int,
        output_channels: int,
        # --- CompactTFT ---
        tft_d_model: int = 64,
        tft_heads: int = 4,
        tft_layers: int = 2,
        tft_dropout: float = 0.3,
        # --- TSMixer ---
        num_blocks: int = 4,
        mixer_dropout: float = 0.3,
        ff_dim: int = 128,
        # --- Graph ---
        num_nodes: int = 6,
        graph_hidden_dim: int = 32,
    ):
        super().__init__()

        # ========== 前处理：CompactTFT ==========
        self.tft = CompactTFT(
            enc_in=input_channels,
            d_model=tft_d_model,
            n_heads=tft_heads,
            n_layers=tft_layers,
            dropout=tft_dropout,
        )

        # TFT 输出维度 → TSMixer 输入通道
        self.tsmixer = TSMixer(
            sequence_length=sequence_length,
            prediction_length=prediction_length,
            input_channels=tft_d_model,
            output_channels=output_channels,
            dropout_rate=mixer_dropout,
            ff_dim=ff_dim,
            num_blocks=num_blocks,
        )

        # ========== 后处理：Graph Residual ==========
        self.graph_block = GraphResidualBlock(
            num_nodes=num_nodes,
            hidden_dim=graph_hidden_dim,
        )

        self.pred_len = prediction_length

    def forward(self, x, static_emb=None):
        """
        x: [B, T, C]
        static_emb: [B, static_dim] or None
        """

        # ---- 1. TFT 编码（时序语义增强）----
        x = self.tft(x, static_emb=static_emb)   # [B, T, d_model]

        # ---- 2. TSMixer（时间-通道混合 + 预测）----
        x = self.tsmixer(x)                      # [B, pred_len, out_channels]

        # ---- 3. squeeze pred_len ----
        if self.pred_len == 1:
            x = x.squeeze(1)                     # [B, out_channels]

        # ---- 4. 图残差（空间一致性）----
        x = self.graph_block(x)                  # [B, out_channels]

        return x



# --- 子模块 ---
class GRN(nn.Module):
    """Gated Residual Network (简化版)"""
    def __init__(self, input_dim, hidden_dim, output_dim=None, dropout=0.3):
        super().__init__()
        output_dim = output_dim or input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(output_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = F.gelu(self.fc1(x))
        x = self.dropout(self.fc2(x))
        gate = torch.sigmoid(self.gate(x))
        x = gate * x + (1 - gate) * residual
        return self.norm(x)
    

class PositionalEncoding(nn.Module):
    """标准 Transformer 位置编码"""
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# --- 主体模块 ---
class CompactTFT(nn.Module):
    """
    改进版 TFT，用于替代 SimpleTFT
    """
    def __init__(self, enc_in, d_model=64, n_heads=4, n_layers=2, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(enc_in, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        # 层堆叠
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*2, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 静态融合 + GRN
        self.static_fuse = GRN(d_model, d_model*2, d_model, dropout=dropout)

        # 输出层归一化
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, static_emb=None):
        """
        x: [B, T, enc_in]
        static_emb: [B, static_dim] -> 融合为 [B, 1, d_model]
        """
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)

        if static_emb is not None:
            s = static_emb.unsqueeze(1).repeat(1, x.size(1), 1)
            x = self.static_fuse(x + s)

        return self.norm(x)


class TSMixer(nn.Module):
    """TSMixer model for time series forecasting.

    This model uses a series of mixer layers to process time series data,
    followed by a linear transformation to project the output to the desired
    prediction length.

    Attributes:
        mixer_layers: Sequential container of mixer layers.
        temporal_projection: Linear layer for temporal projection.

    Args:
        sequence_length: Length of the input time series sequence.
        prediction_length: Desired length of the output prediction sequence.
        input_channels: Number of input channels.
        output_channels: Number of output channels. Defaults to None.
        activation_fn: Activation function to use. Defaults to "relu".
        num_blocks: Number of mixer blocks. Defaults to 2.
        dropout_rate: Dropout rate for regularization. Defaults to 0.1.
        ff_dim: Dimension of feedforward network inside mixer layer. Defaults to 64.
        normalize_before: Whether to apply layer normalization before or after mixer layer.
        norm_type: Type of normalization to use. "batch" or "layer". Defaults to "batch".
    """

    def __init__(
        self,
        sequence_length: int,
        prediction_length: int,
        input_channels: int,
        output_channels: int = None,
        activation_fn: str = "relu",
        num_blocks: int = 2,
        dropout_rate: float = 0.1,
        ff_dim: int = 64,
        normalize_before: bool = True,
        norm_type: str = "batch",
    ):
        super().__init__()

        # Transform activation_fn to callable
        activation_fn = getattr(F, activation_fn)

        # Transform norm_type to callable
        assert norm_type in {
            "batch",
            "layer",
        }, f"Invalid norm_type: {norm_type}, must be one of batch, layer."
        norm_type = TimeBatchNorm2d if norm_type == "batch" else nn.LayerNorm

        # Build mixer layers
        self.mixer_layers = self._build_mixer(
            num_blocks,
            input_channels,
            output_channels,
            ff_dim=ff_dim,
            activation_fn=activation_fn,
            dropout_rate=dropout_rate,
            sequence_length=sequence_length,
            normalize_before=normalize_before,
            norm_type=norm_type,
        )

        # Temporal projection layer
        self.temporal_projection = nn.Linear(sequence_length, prediction_length)

    def _build_mixer(
        self, num_blocks: int, input_channels: int, output_channels: int, **kwargs
    ):
        """Build the mixer blocks for the model.

        Args:
            num_blocks (int): Number of mixer blocks to be built.
            input_channels (int): Number of input channels for the first block.
            output_channels (int): Number of output channels for the last block.
            **kwargs: Additional keyword arguments for mixer layer configuration.

        Returns:
            nn.Sequential: Sequential container of mixer layers.
        """
        output_channels = output_channels if output_channels is not None else input_channels
        channels = [input_channels] * (num_blocks - 1) + [output_channels]

        return nn.Sequential(
            *[
                MixerLayer(input_channels=in_ch, output_channels=out_ch, **kwargs)
                for in_ch, out_ch in zip(channels[:-1], channels[1:])
            ]
        )

    def forward(self, x_hist: torch.Tensor) -> torch.Tensor:
        """Forward pass of the TSMixer model.

        Args:
            x_hist (torch.Tensor): Input time series tensor.

        Returns:
            torch.Tensor: The output tensor after processing by the model.
        """
        x = self.mixer_layers(x_hist)

        x_temp = feature_to_time(x)
        x_temp = self.temporal_projection(x_temp)
        x = time_to_feature(x_temp)

        return x



class GraphResidualBlock(nn.Module):
    def __init__(self, num_nodes, hidden_dim=32):
        """
        num_nodes: 节点数 (例如6)
        hidden_dim: 中间隐藏维度
        """
        super().__init__()
        self.num_nodes = num_nodes

        # 可学习邻接矩阵 [N, N]
        self.A_param = nn.Parameter(torch.randn(num_nodes, num_nodes))

        # 节点特征变换
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x: [B, N] 或 [B, 1, N]
        """
        if x.dim() == 3:  # [B, 1, N]
            x = x.squeeze(1)  # [B, N]

        # 归一化邻接矩阵（按行 softmax，保证权重和为1）
        A = torch.softmax(self.A_param, dim=-1)  # [N, N]

        h = x.unsqueeze(-1)         # [B, N, 1]
        h = self.fc1(h)             # [B, N, H]
        h = torch.matmul(A, h)      # [B, N, H]
        h = F.relu(h)
        h = self.fc2(h).squeeze(-1) # [B, N]

        return x + h  # 残差连接
