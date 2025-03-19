import torch
from torch import nn
from timm.models.layers import DropPath
import math

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=3):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.norm =nn.BatchNorm1d(channel)

    def forward(self, x):

        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        x_channel = x * y.expand_as(x)
        x = self.norm(x + x_channel)

        return x

class AttentionBlock(nn.Module):
    def __init__(self, in_channel: int):
        super().__init__()
        # Spatial Attention Module
        self.conv_1 = nn.Conv1d(in_channel, in_channel, kernel_size=1, stride=1, padding='same')
        self.max_pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.linear = nn.Conv1d(in_channel * 2, in_channel, kernel_size=1, stride=1, padding='same')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_1(x)
        res_x = x
        x_max = self.max_pool(x)
        x_avg = self.avg_pool(x)
        x = torch.cat([x_max, x_avg], dim=1)
        x = self.linear(x)
        x = self.sigmoid(x)
        x = res_x * x
        return x


class ECABlock(nn.Module):
    """ECA attention"""
    def __init__(self, channel: int):
        super(ECABlock, self).__init__()

        k_size = math.ceil(math.log(channel, 2) / 2 + 0.5)

        k_size = int(k_size)
        if k_size % 2 == 0:
            k_size += 1

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()
        self.norm =nn.BatchNorm1d(channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = y.transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.transpose(-1, -2)
        x_channel = x * y
        x = self.norm(x + x_channel)
        return x


class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.active = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.active(x)
        x = self.fc2(x)
        return x



class GroupChannelAttention(nn.Module):

    def __init__(self, dim, num_heads=4,
                 qkv_bias=True,
                 ffn=True, mlp_ratio=3,
                 drop_path = 0.05,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.ffn = ffn
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.ffn:
            self.norm = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        k = k * self.scale
        attention = k.transpose(-1, -2) @ v
        attention = attention.softmax(dim=-1)
        x_feature = (attention @ q.transpose(-1, -2)).transpose(-1, -2)
        x_feature = x_feature.transpose(1, 2).reshape(B, N, C)
        x = x + self.drop_path(x_feature)
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm(x)))
        return x


class MDA_Net(nn.Module):
    def __init__(self, in_channel=1, out_channel=5,
                 dilation=[1,3,5],kernel_size = [32,64],
                 num_heads = [6,2], drop_path = 0.05):
        super(MDA_Net, self).__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        print(self.dilation,self.kernel_size,self.num_heads)

        self.layer1_1_1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=16, kernel_size=self.kernel_size[0], stride=1, dilation =self.dilation[0],padding='same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.layer1_1_2 = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=16, kernel_size=self.kernel_size[0], stride=1, dilation =self.dilation[1],padding='same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.layer1_1_3 = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=16, kernel_size=self.kernel_size[0], stride=1, dilation =self.dilation[2], padding='same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.layer1_2_1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=16, kernel_size=self.kernel_size[1], stride=1,dilation =self.dilation[0], padding='same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.layer1_2_2 = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=16, kernel_size=self.kernel_size[1], stride=1,dilation =self.dilation[1], padding='same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.layer1_2_3 = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=16, kernel_size=self.kernel_size[1], stride=1, dilation =self.dilation[2], padding='same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=96, out_channels=48, kernel_size=(64,), stride=(16,)),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )

        #尝试局部和通道的双注意力，最后只用了GroupChannelAttention
        self.encoder_layer_1 = nn.TransformerEncoderLayer(d_model=96, nhead=4, batch_first=True, dim_feedforward=128)
        self.transformer_encoder1 = nn.TransformerEncoder(self.encoder_layer_1, num_layers=1)
        self.GroupChannelAttention = GroupChannelAttention(dim=96, num_heads=self.num_heads[0], qkv_bias=True)
        self.SeChannelAttention = SEBlock(channel=96)
        self.GroupChannelAttention2 = GroupChannelAttention(dim=48, num_heads=self.num_heads[1], qkv_bias=True)
        self.encoder_layer_2 = nn.TransformerEncoderLayer(d_model=96, nhead=4, batch_first=True, dim_feedforward=128)
        self.transformer_encoder2 = nn.TransformerEncoder(self.encoder_layer_2, num_layers=1)
        self.SeChannelAttention2 = SEBlock(channel=96)

        self.fc = nn.Sequential(
            nn.Linear(5376, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, out_channel),
        )

    def forward(self, x):
        x1_1_1 = self.layer1_1_1(x)
        x1_1_2 = self.layer1_1_2(x)
        x1_1_3 = self.layer1_1_3(x)
        x1_1= torch.cat((x1_1_1, x1_1_2, x1_1_3), dim=1)
        x1_2_1 = self.layer1_2_1(x)
        x1_2_2 = self.layer1_2_2(x)
        x1_2_3 = self.layer1_2_3(x)
        x1_2 = torch.cat((x1_2_1, x1_2_2, x1_2_3), dim=1)
        x1 = torch.cat((x1_1, x1_2), dim=1)
        x_Channel = self.GroupChannelAttention(x1.permute(0, 2, 1))
        x_Channel = self.pool(x_Channel.permute(0, 2, 1))
        x = self.layer2(x_Channel)
        x = x.permute(0, 2, 1)
        x = self.GroupChannelAttention2(x)
        x = x.permute(0, 2, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


#对比其他的通道Attention Block
class MDA_Net_Attention(nn.Module):

    def __init__(self, in_channel=1, out_channel=5,
                 dilation=[1,3,5],kernel_size = [32,64],
                 num_heads = [6,2],ChannelAtten = None,
                 drop_path = 0.05):
        super(MDA_Net_Attention, self).__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.ChannelAtten = ChannelAtten
        print(self.dilation,self.kernel_size,self.num_heads,self.ChannelAtten)
        self.layer1_1_1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=16, kernel_size=self.kernel_size[0], stride=1, dilation =self.dilation[0],padding='same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.layer1_1_2 = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=16, kernel_size=self.kernel_size[0], stride=1, dilation =self.dilation[1],padding='same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.layer1_1_3 = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=16, kernel_size=self.kernel_size[0], stride=1, dilation =self.dilation[2], padding='same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.layer1_2_1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=16, kernel_size=self.kernel_size[1], stride=1,dilation =self.dilation[0], padding='same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.layer1_2_2 = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=16, kernel_size=self.kernel_size[1], stride=1,dilation =self.dilation[1], padding='same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.layer1_2_3 = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=16, kernel_size=self.kernel_size[1], stride=1, dilation =self.dilation[2], padding='same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=96, out_channels=48, kernel_size=(64,), stride=(16,)),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )

        if self.ChannelAtten == 'SeBlock':
            self.ChannelAttention = SEBlock(channel=96)
            self.ChannelAttention2 = SEBlock(channel=48)

        if self.ChannelAtten == 'ECA_Block':
            self.ChannelAttention = ECABlock(channel=96)
            self.ChannelAttention2 = ECABlock(channel=48)

        if self.ChannelAtten == 'AttentionBlock':
            self.ChannelAttention = AttentionBlock(in_channel=96)
            self.ChannelAttention2 = AttentionBlock(in_channel=48)

        self.fc = nn.Sequential(
            nn.Linear(5376, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, out_channel),

        )

    def forward(self, x):
        x1_1_1 = self.layer1_1_1(x)
        x1_1_2 = self.layer1_1_2(x)
        x1_1_3 = self.layer1_1_3(x)
        x1_1= torch.cat((x1_1_1, x1_1_2, x1_1_3), dim=1)
        x1_2_1 = self.layer1_2_1(x)
        x1_2_2 = self.layer1_2_2(x)
        x1_2_3 = self.layer1_2_3(x)
        x1_2 = torch.cat((x1_2_1, x1_2_2, x1_2_3), dim=1)
        x1 = torch.cat((x1_1, x1_2), dim=1)

        if self.ChannelAtten is None:
            x = self.pool(x1)
            x = self.layer2(x)

        else:
            x_Channel = self.ChannelAttention(x1)
            x_Channel = self.pool(x_Channel)
            x = self.layer2(x_Channel)
            x = self.ChannelAttention2(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


import torchinfo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X = torch.randn(1, 1 ,3699).to(device)
model = MDA_Net(in_channel=1, out_channel=5).to(device)
torchinfo.summary(model, input_size=(1, 1, 3699))
output = model(X)
print(output.shape)