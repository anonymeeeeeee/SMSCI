import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader 
import torchvision
from torchvision import datasets 
from torchvision.transforms import ToTensor 
from torchvision.utils import save_image
from torch.distributions.normal import Normal
from torchvision import datasets, models, transforms
from torchvision.transforms import ToTensor 
import numpy as np
import time
import yaml

global dropout_val
dropout_val             = 0.2 
T_obs                   = 8
T_pred                  = 12
T_total                 = T_obs + T_pred
data_id                 = 0 
batch_size              = 64
chunk_size              = batch_size * T_total # Chunksize should be multiple of T_total
in_size = 2
stochastic_out_size     = in_size * 2
teacher_forcing_ratio   = 0.7 
regularization_factor   = 0.5
avg_n_path_eval         = 20
bst_n_path_eval         = 20
startpoint_mode         = "off"
enc_out                 = "on"
biased_loss_mode        = 0 # 0 , 1
table       = "dataset_T_length_"+str(T_total)+"delta_coordinates"
epoch_num   = 200
from_epoch  = 0
# Visual Variables
image_size              = 256  
image_dimension         = 3
mask_size               = 16
visual_features_size    = 128 
visual_embed_size       = 64 
vsn_module_out_size    = 256

# ------------------------------------------------------------------------------
# Initialize random weights for NN models
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.2, 0.2)

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)

def get_noise(shape, noise_type):
    if noise_type == 'gaussian':
        return torch.randn(*shape).cuda()
    elif noise_type == 'uniform':
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)

class Encoder(nn.Module):
    """Encoder is part of both TrajectoryGenerator and TrajectoryDiscriminator"""
    def __init__(self, embedding_dim=64, h_dim=256, mlp_dim=1024, num_layers=1,dropout=0.2):
        super(Encoder, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)
        self.dropout                    = nn.Dropout(dropout)
        self.embedder_out               = nn.Sequential(
                                                        nn.Linear(T_obs*h_dim, h_dim),
                                                        nn.ReLU(),
                                                        nn.Dropout(p=dropout),
                                                        nn.Linear(h_dim, h_dim),
                                                        nn.ReLU()
                                                        )     

    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
            torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        )
    
    def emb_out(self,input):
        out= self.embedder_out(input)
        return out

    def forward(self, obs_traj):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        # Encode observed Trajectory
        batch = obs_traj.size(1)
        obs_traj_embedding = self.spatial_embedding(obs_traj.reshape(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(-1, batch, self.embedding_dim)
        state_tuple = self.init_hidden(batch)
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        final_h = state[0]
        return final_h, output, state # Sortie de Gan + Sorties d'Introvert

# Dynamic Context Encoder
def Resnet(pretrain=True,layers_to_unfreeze=8,layers_to_delete=2,in_planes=3):
    """
    param:
        pretrain: Define if we load a pretrained model from ImageNet
        layers_to_unfreeze: Define the number of layers that we want to train at the end of the Resnet
        layers_to_delete: Define the numbers of layers that we want to delete
        in_planes: Define the numbers of input channels of images (supported values: 1,2 or 3)
    return: The Resnet model
    """
    resnet = torchvision.models.resnet18(pretrained=pretrain)
    model = nn.Sequential()
    number_of_layers = len(list(resnet.children())) - layers_to_delete

    if number_of_layers<layers_to_unfreeze:
        layers_to_unfreeze = number_of_layers
    layers_to_freeze = number_of_layers - layers_to_unfreeze
    i=0
    for child in resnet.children():
        if i==0 and in_planes<3:
            if i<layers_to_freeze: # Define if we freeze this layer or no
                for param in child.parameters():
                    param.requires_grad = False # Freeze the layers by passing requires_grad attribute to False
            w = child._parameters['weight'].data # Get the weight for 3 channels data
            child._modules['0'] = nn.Conv2d(in_planes, 64, kernel_size=3, padding=1) # Define the new conv layer
            if in_planes == 1:
                child._parameters['weight'].data = w.mean(dim=1, keepdim=True) # If the number of channels is 1 we made the mean of channels to set the new weight
            else:
                child._parameters['weight'].data = w[:, :-1] * 1.5

        if i<layers_to_freeze: # Define if we freeze this layer or no
            for param in child.parameters():
                param.requires_grad = False # Freeze the layers by passing requires_grad attribute to False
        if i<number_of_layers: # To define if we keep this layer or not
            model.append(child) 
        i+=1
    return model

def position_embedding(input, d_model):
    input = input.view(-1, 1)
    dim = torch.arange(d_model // 2, dtype=torch.float32, device=input.device).view(1, -1)
    sin = torch.sin(input / 10000 ** (2 * dim / d_model))
    cos = torch.cos(input / 10000 ** (2 * dim / d_model))
    out = torch.zeros((input.shape[0], d_model), device=input.device)
    out[:, ::2] = sin
    out[:, 1::2] = cos
    return out

def sinusoid_encoding_table(max_len, d_model):
    pos = torch.arange(max_len, dtype=torch.float32)
    out = position_embedding(pos, d_model)
    return out

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h):
        """
        param:
        d_model: Output dimensionality of the model
        d_k: Dimensionality of queries and keys
        d_v: Dimensionality of values
        h: Number of heads
        """
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights(gain=1.0)

    def init_weights(self, gain=1.0):
        nn.init.xavier_normal_(self.fc_q.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_k.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_v.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_o.weight, gain=gain)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values):
        """
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :return:
        """
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        att = torch.softmax(att, -1)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out

    
#Multi-head attention layer with Dropout and Layer Normalization
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dff=2048, dropout=.1):
        super(MultiHeadAttention, self).__init__()
        self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(*[nn.Linear(d_model, dff), nn.ReLU(inplace=True), nn.Dropout(p=dropout),nn.Linear(dff, d_model)])

    def forward(self, queries, keys, values):
        att = self.attention(queries, keys, values)
        att = self.dropout(att)
        att = self.fc(att)
        att = self.dropout(att)
        return self.layer_norm(queries + att)

class EncoderSelfAttention(nn.Module):
    def __init__(self,device,d_model, d_k, d_v, n_head, dff=2048, dropout_transformer=.1, n_module=6):
        super(EncoderSelfAttention, self).__init__()
        self.encoder = nn.ModuleList([MultiHeadAttention(d_model, d_k, d_v, n_head, dff, dropout_transformer) for _ in range(n_module)])
        self.device = device
    
    def forward(self, x):
        in_encoder = x + sinusoid_encoding_table(x.shape[1], x.shape[2]).expand(x.shape).to(self.device)
        for l in self.encoder:
            in_encoder = l(in_encoder, in_encoder, in_encoder)
        return in_encoder


class features_extraction(nn.Module):
    """
    param:
    conv_model: The convolution model used before capsules for the moment only ResNet is supported
    in_planes: Numbers of channels for the image
    """
    def __init__(self,conv_model,in_planes: int):
        super().__init__()
        self.conv_model = conv_model
        self.in_planes = in_planes
        self.pooling = nn.AdaptiveAvgPool2d((1,1))
    def forward(self,input):
        shape = input.size()
        x = input.reshape(-1,self.in_planes,shape[-2],shape[-1]) #view
        x = self.conv_model(x)
        x = self.pooling(x)
        return x
    
# Dynamic Context Encoder
class _DCE_Transformer(nn.Module):
    """Multi-Modal model on 3 or 1 channel"""
    def __init__(self,device,backbone="resnet",in_planes=3,pretrained= True,input_dim=512,layers_to_unfreeze=8,layers_to_delete=2,n_head=8,n_module=6,ff_size=1024,dropout1d=0.5):
        super(_DCE_Transformer, self).__init__()

        self.in_planes = in_planes
        self.device = device
        self.conv_name = backbone
        self.conv_model = None
        
        if self.conv_name.lower()=="resnet":
            self.conv_model = Resnet(pretrained,layers_to_unfreeze,layers_to_delete,in_planes)
        else:
            raise NotImplementedError("The model {} is not supported!".format(self.conv_name))
        self.conv_model.to(device)
        self.features = features_extraction(self.conv_model,in_planes)

        self.self_attention = EncoderSelfAttention(device,input_dim,64,64,n_head=n_head,dff=ff_size,dropout_transformer=dropout1d,n_module=n_module)

        self.pool = nn.AdaptiveAvgPool2d((1,input_dim)) #final pooling

    def forward(self, x):
        shape = x.shape
        x = self.features(x)
        x = x.view(shape[0],shape[1],-1)
        x = self.self_attention(x)
        return x
    

class Decoder(nn.Module):
    """Decoder is part of TrajectoryGenerator"""
    def __init__(
        self, seq_len, embedding_dim=64, h_dim=512, mlp_dim=1024, num_layers=1,
        pool_every_timestep=True, dropout=0.2, bottleneck_dim=1024,
        activation='relu', batch_norm=True, pooling_type='pool_net',
        neighborhood_size=2.0, grid_size=8
    ):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.pool_every_timestep = pool_every_timestep
        self.stochastic_out_size = stochastic_out_size

        
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        
        if startpoint_mode=="on":
            self.decoder                    = nn.LSTM(embedding_dim, h_dim + in_size, num_layers, dropout=dropout, batch_first=True)
            self.fC_mu                      = nn.Sequential(
                                                            nn.Linear(h_dim + in_size, int(h_dim/4), bias=True),
                                                            nn.ReLU(),
                                                            nn.Dropout(p=dropout),
                                                            nn.Linear(int(h_dim/4), self.stochastic_out_size, bias=True)
                                                            )
        else:
            self.decoder                    = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout, batch_first=True)
            self.fC_mu                      = nn.Sequential(nn.Linear(h_dim , int(h_dim/4), bias=True),nn.ReLU(),nn.Dropout(p=dropout),nn.Linear(int(h_dim/4), self.stochastic_out_size, bias=True))
        self.dropout                        = nn.Dropout(dropout)

        self.reducted_size = int((h_dim//2-1)/3)+1
        
        if startpoint_mode =="on":
            self.reducted_size2 = int((h_dim//2 -1)/3)+1
            self.FC_dim_red                     = nn.Sequential(
                                                            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
                                                            nn.Flatten(start_dim=1, end_dim=-1),
                                                            nn.Linear(self.reducted_size*self.reducted_size2, h_dim+in_size, bias=True),
                                                            nn.ReLU()
                                                            )
        else:
            self.FC_dim_red                     = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=3, padding=1),nn.Flatten(start_dim=1, end_dim=-1),nn.Linear(self.reducted_size*self.reducted_size, h_dim, bias=True),nn.ReLU())

        if pool_every_timestep:
            if pooling_type == 'pool_net':
                self.pool_net = PoolHiddenNet(
                    embedding_dim=self.embedding_dim,
                    h_dim=self.h_dim,
                    mlp_dim=mlp_dim,
                    bottleneck_dim=bottleneck_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout
                )
            elif pooling_type == 'attpool':
                self.pool_net = PoolHiddenAttNet( 
                    embedding_dim=self.embedding_dim,
                    h_dim=self.h_dim,
                    mlp_dim=mlp_dim,
                    bottleneck_dim=bottleneck_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    self_attention_dim=self_attention_dim
                )
            elif pooling_type == 'spool':
                self.pool_net = SocialPooling(
                    h_dim=self.h_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    neighborhood_size=neighborhood_size,
                    grid_size=grid_size
                )

            mlp_dims = [h_dim + bottleneck_dim, mlp_dim, h_dim]
            self.mlp = make_mlp(
                mlp_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

        self.hidden2pos = nn.Linear(h_dim, 2)
        

    def dim_red(self, input):
        output = self.FC_dim_red(input)
        return output

    def forward(self, input, hidden): 
        embedding                       = self.spatial_embedding(input.reshape(-1,2))
        embedding                       = F.relu(self.dropout(embedding))
        output, hidden                  = self.decoder(embedding.unsqueeze(1), ( hidden[0],hidden[1] ))
        prediction                      = self.fC_mu(output.squeeze(0)) 
        return prediction, hidden
    
#This model incorporates a self-attention mechanism. It calculates the self-attention weights for the relative position embeddings.
#After obtaining the self-attention output, it concatenates this output with the original relative position embeddings before passing them through the MLP network. 
#Similar to PoolHiddenNet, it also uses max-pooling to get the final output.
#The inclusion of self-attention in enables the model to learn dependencies and relationships between relative position embeddings more effectively.
#By incorporating self-attention, the model can focus on relevant parts of the input sequences, potentially improving the overall performance by capturing more complex and long-range dependencies among the input elements.
class PoolHiddenAttNet(nn.Module):
    """Pooling module as proposed in our paper"""
    def __init__(self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,activation='relu', batch_norm=True, dropout=0.0, self_attention_dim=128):
        super(PoolHiddenAttNet, self).__init__()
        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim
        self.self_attention_dim = self_attention_dim
        mlp_pre_dim = 2080
        mlp_pre_pool_dims = [mlp_pre_dim, 512, bottleneck_dim]
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.mlp_pre_pool = make_mlp(mlp_pre_pool_dims,activation=activation,batch_norm=batch_norm,dropout=dropout)

        self.self_attention = nn.MultiheadAttention(embed_dim=self_attention_dim, num_heads=1)

    def repeat(self, tensor, num_reps):
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_end_pos = end_pos[start:end]
            
            curr_hidden_1 = curr_hidden.repeat(num_ped, 1)
            curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1)
            curr_end_pos_2 = self.repeat(curr_end_pos, num_ped)
            curr_rel_pos = curr_end_pos_1 - curr_end_pos_2
            curr_rel_embedding = self.spatial_embedding(curr_rel_pos)

            self_attention_input = curr_rel_embedding.unsqueeze(0).permute(1, 0, 2)
            self_attention_output, _ = self.self_attention(self_attention_input,self_attention_input, self_attention_input)
            self_attention_output = self_attention_output.permute(0,2,1)
            self_attention_output_squeezed = torch.squeeze(self_attention_output, dim=2)  
            curr_rel_embedding = torch.cat([curr_rel_embedding, self_attention_output_squeezed], dim=1) 
            mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)
            curr_pool_h = self.mlp_pre_pool(mlp_h_input)
            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0]
            pool_h.append(curr_pool_h)

        pool_h = torch.cat(pool_h, dim=0)
        return pool_h

# This suggestion does not include any self-attention mechanism. It performs concatenation of relative position embeddings and hidden states and processes them through an MLP network.
class PoolHiddenNet(nn.Module):
    def __init__(self, embedding_dim=64, h_dim=256, mlp_dim=1024, bottleneck_dim=1024,activation='relu', batch_norm=True, dropout=0.0):
        super(PoolHiddenNet, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim

        mlp_pre_dim = embedding_dim + h_dim
        mlp_pre_pool_dims = [mlp_pre_dim, 512, bottleneck_dim]

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.mlp_pre_pool = make_mlp(mlp_pre_pool_dims,activation=activation,batch_norm=batch_norm,dropout=dropout)

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        """
        Inputs:
        - h_states: Tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        - end_pos: Tensor of shape (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, bottleneck_dim)
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_end_pos = end_pos[start:end]
            # Repeat -> H1, H2, H1, H2
            curr_hidden_1 = curr_hidden.repeat(num_ped, 1)
            # Repeat position -> P1, P2, P1, P2
            curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1)
            # Repeat position -> P1, P1, P2, P2
            curr_end_pos_2 = self.repeat(curr_end_pos, num_ped)
            curr_rel_pos = curr_end_pos_1 - curr_end_pos_2
            curr_rel_embedding = self.spatial_embedding(curr_rel_pos),

            mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)
        
            curr_pool_h = self.mlp_pre_pool(mlp_h_input)
            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0]
            pool_h.append(curr_pool_h)
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h

class SocialPooling(nn.Module):
    """Current state of the art pooling mechanism of Social_LSTM"""
    def __init__(self, h_dim=256, activation='relu', batch_norm=True, dropout=0.0,neighborhood_size=2.0, grid_size=8, pool_dim=None):
        super(SocialPooling, self).__init__()
        self.h_dim = h_dim
        self.grid_size = grid_size
        self.neighborhood_size = neighborhood_size
        if pool_dim:
            mlp_pool_dims = [grid_size * grid_size * h_dim, pool_dim]
        else:
            mlp_pool_dims = [grid_size * grid_size * h_dim, h_dim]

        self.mlp_pool = make_mlp(
            mlp_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )

    def get_bounds(self, ped_pos):
        top_left_x = ped_pos[:, 0] - self.neighborhood_size / 2
        top_left_y = ped_pos[:, 1] + self.neighborhood_size / 2
        bottom_right_x = ped_pos[:, 0] + self.neighborhood_size / 2
        bottom_right_y = ped_pos[:, 1] - self.neighborhood_size / 2
        top_left = torch.stack([top_left_x, top_left_y], dim=1)
        bottom_right = torch.stack([bottom_right_x, bottom_right_y], dim=1)
        return top_left, bottom_right

    def get_grid_locations(self, top_left, other_pos):
        cell_x = torch.floor(
            ((other_pos[:, 0] - top_left[:, 0]) / self.neighborhood_size) *
            self.grid_size)
        cell_y = torch.floor(
            ((top_left[:, 1] - other_pos[:, 1]) / self.neighborhood_size) *
            self.grid_size)
        grid_pos = cell_x + cell_y * self.grid_size
        return grid_pos

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        """
        Inputs:
        - h_states: Tesnsor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - end_pos: Absolute end position of obs_traj (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, h_dim)
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            grid_size = self.grid_size * self.grid_size
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_hidden_repeat = curr_hidden.repeat(num_ped, 1)
            curr_end_pos = end_pos[start:end]
            curr_pool_h_size = (num_ped * grid_size) + 1
            curr_pool_h = curr_hidden.new_zeros((curr_pool_h_size, self.h_dim))
            # curr_end_pos = curr_end_pos.data
            top_left, bottom_right = self.get_bounds(curr_end_pos)

            # Repeat position -> P1, P2, P1, P2
            curr_end_pos = curr_end_pos.repeat(num_ped, 1)
            # Repeat bounds -> B1, B1, B2, B2
            top_left = self.repeat(top_left, num_ped)
            bottom_right = self.repeat(bottom_right, num_ped)

            grid_pos = self.get_grid_locations(
                    top_left, curr_end_pos).type_as(seq_start_end)
            # Make all positions to exclude as non-zero
            # Find which peds to exclude
            x_bound = ((curr_end_pos[:, 0] >= bottom_right[:, 0]) +
                       (curr_end_pos[:, 0] <= top_left[:, 0]))
            y_bound = ((curr_end_pos[:, 1] >= top_left[:, 1]) +
                       (curr_end_pos[:, 1] <= bottom_right[:, 1]))

            within_bound = x_bound + y_bound
            within_bound[0::num_ped + 1] = 1  # Don't include the ped itself
            within_bound = within_bound.view(-1)

            grid_pos += 1
            total_grid_size = self.grid_size * self.grid_size
            offset = torch.arange(0, total_grid_size * num_ped, total_grid_size).type_as(seq_start_end)

            offset = self.repeat(offset.view(-1, 1), num_ped).view(-1)
            grid_pos += offset
            grid_pos[within_bound != 0] = 0
            grid_pos = grid_pos.view(-1, 1).expand_as(curr_hidden_repeat)

            curr_pool_h = curr_pool_h.scatter_add(0, grid_pos,curr_hidden_repeat)
            curr_pool_h = curr_pool_h[1:]
            pool_h.append(curr_pool_h.view(num_ped, -1))

        pool_h = torch.cat(pool_h, dim=0)
        pool_h = self.mlp_pool(pool_h)
        return pool_h

class TrajectoryGenerator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=256,
        decoder_h_dim=128, mlp_dim=1024, num_layers=1, noise_dim=(0, ),
        noise_type='gaussian', noise_mix_type='ped', pooling_type='pool_net',
        pool_every_timestep=True, dropout=0.2, bottleneck_dim=1024,
        activation='relu', batch_norm=True, neighborhood_size=2.0, grid_size=8, self_attention_dim=1024
    ):
        super(TrajectoryGenerator, self).__init__()

        if pooling_type and pooling_type.lower() == 'none':
            pooling_type = None

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.mlp_dim = mlp_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim
        self.num_layers = num_layers
        self.noise_type = noise_type
        self.noise_mix_type = noise_mix_type
        self.pooling_type = pooling_type
        self.noise_first_dim = 0
        self.pool_every_timestep = pool_every_timestep
        self.bottleneck_dim = 1024
        self.self_attention_dim= self_attention_dim
        
        self.encoder = Encoder(embedding_dim=embedding_dim,h_dim=encoder_h_dim,mlp_dim=mlp_dim,num_layers=num_layers,dropout=dropout)
        self.encoder.apply(init_weights)
        self.vision = _DCE_Transformer(device,dropout1d=dropout_val)
        self.vision.apply(init_weights)
        self.pooling = nn.AdaptiveAvgPool1d(256)
        self.decoder = Decoder(
            pred_len,
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            pool_every_timestep=pool_every_timestep,
            dropout=dropout,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            batch_norm=batch_norm,
            pooling_type=pooling_type,
            grid_size=grid_size,
            neighborhood_size=neighborhood_size
        )
        self.decoder.apply(init_weights)
        
        if device.type=='cuda':
            self.encoder.cuda()
            self.decoder.cuda()
            self.vision.cuda()
        
        if pooling_type == 'pool_net':
            self.pool_net = PoolHiddenNet(
                embedding_dim=self.embedding_dim,
                h_dim=encoder_h_dim,
                mlp_dim=mlp_dim,
                bottleneck_dim=bottleneck_dim,
                activation=activation,
                batch_norm=batch_norm
            )
        elif pooling_type == 'attpool':
            self.pool_net = PoolHiddenAttNet( 
                embedding_dim=self.embedding_dim,
                h_dim=encoder_h_dim,
                mlp_dim=mlp_dim,
                bottleneck_dim=bottleneck_dim,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
                self_attention_dim=self_attention_dim
            )
        elif pooling_type == 'spool':
            self.pool_net = SocialPooling(
                h_dim=encoder_h_dim,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
                neighborhood_size=neighborhood_size,
                grid_size=grid_size
            )

        if self.noise_dim != None: 
          if self.noise_dim[0] == 0:
            self.noise_dim = None
          else:
            self.noise_first_dim = noise_dim[0]

        # Decoder Hidden
        if pooling_type:
            input_dim = encoder_h_dim + bottleneck_dim
        else:
            input_dim = encoder_h_dim

        if self.mlp_decoder_needed():
            mlp_decoder_context_dims = [input_dim, mlp_dim, decoder_h_dim - self.noise_first_dim]

            self.mlp_decoder_context = make_mlp(mlp_decoder_context_dims,activation=activation,batch_norm=batch_norm,dropout=dropout)

    def add_noise(self, _input, seq_start_end, user_noise=None):
        """
        Inputs:
        - _input: Tensor of shape (_, decoder_h_dim - noise_first_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Outputs:
        - decoder_h: Tensor of shape (_, decoder_h_dim)
        """
        if not self.noise_dim:
            return _input

        if self.noise_mix_type == 'global':
            noise_shape = (seq_start_end.size(0), ) + self.noise_dim
        else:
            noise_shape = (_input.size(0), ) + self.noise_dim

        if user_noise is not None:
            z_decoder = user_noise
        else:
            z_decoder = get_noise(noise_shape, self.noise_type)

        if self.noise_mix_type == 'global':
            _list = []
            for idx, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                _vec = z_decoder[idx].view(1, -1)
                _to_cat = _vec.repeat(end - start, 1)
                _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
            decoder_h = torch.cat(_list, dim=0)
            return decoder_h

        decoder_h = torch.cat([_input, z_decoder], dim=1)

        return decoder_h

    def mlp_decoder_needed(self):
        if (
            self.noise_dim or self.pooling_type or
            self.encoder_h_dim != self.decoder_h_dim
        ):
            return True
        else:
            return False

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, visual_input_tensor, output_tensor, user_noise=None):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Output:
        - pred_traj_rel: Tensor of shape (self.pred_len, batch, 2)
        """
        batch = obs_traj_rel.size(1)
        
        final_encoder_h,_,_ = self.encoder(obs_traj)
        encoder_hidden  = self.encoder.init_hidden(batch)
        encoder_outputs = torch.zeros(batch, T_obs, self.encoder_h_dim).cuda()#.cpu()
        start_point     = (obs_traj[0,:,:]).to(device).clone().detach()

        if startpoint_mode=="on":
            obs_traj[0,:,:]    = 0
        
        for t in range(0,T_obs):
            _, encoder_output, encoder_hidden  = self.encoder(torch.transpose(obs_traj[t,:,:],0,1))
            encoder_outputs[:,t,:]          = encoder_output.squeeze(1)

        batch_vis = self.encoder_h_dim
        batch_dec = self.decoder_h_dim 

        # Visual extraction/attention       
        if enc_out=="on" and startpoint_mode=="on":
            print("shape encoder outputs:", encoder_outputs.shape)
            e1= encoder_outputs.view(batch_vis,-1)
            print("shape encoder_outputs.view(batch_vis,-1)):", e1.shape)
            encoder_extract             = self.encoder.emb_out(encoder_outputs.view(batch_vis,-1))
            condition                   = torch.cat([encoder_extract.view(batch_vis,-1),start_point.view(batch_vis,-1)],dim=-1)
            visual_initial_vsn          = self.vision(visual_input_tensor)
            visual_initial_vsn          = self.pooling(visual_initial_vsn)
            decoder_hidden              = [torch.cat([encoder_extract.view(batch_vis,-1),   visual_initial_vsn.view(batch_vis,-1)],dim=-1).unsqueeze(0),\
                                           torch.cat([encoder_hidden[1].view(batch_vis,-1), visual_initial_vsn.view(batch_vis,-1)],dim=-1).unsqueeze(0)]
            
        elif enc_out=="on" and startpoint_mode=="off":  
            encoder_extract             = self.encoder.emb_out(encoder_outputs.view(batch,-1))
            visual_initial_vsn          = self.vision(visual_input_tensor)
            visual_initial_vsn          = self.pooling(visual_initial_vsn)
            decoder_hidden              = [torch.cat([encoder_extract.view(batch_vis,-1),   visual_initial_vsn.view(batch_vis,-1)],dim=-1).unsqueeze(0),\
                                           torch.cat([encoder_hidden[1].view(batch_vis,-1), visual_initial_vsn.view(batch_vis,-1)],dim=-1).unsqueeze(0)]
        elif enc_out=="off" and startpoint_mode=="on":
            condition = torch.cat([encoder_hidden[0].view(batch_size,-1),start_point.view(batch_size,-1)],dim=-1)
            visual_initial_vsn, attn_rep, attn2, attn4, attn6   = self.vision(visual_input_tensor,condition)
            decoder_hidden              = [torch.cat([encoder_hidden[0].view(batch_vis,-1), visual_initial_vsn.view(batch_vis,-1)],dim=-1).unsqueeze(0),\
                                           torch.cat([encoder_hidden[1].view(batch_vis,-1), visual_initial_vsn.view(batch_vis,-1)],dim=-1).unsqueeze(0)]
        else:
            visual_initial_vsn          = self.vision(visual_input_tensor)
            visual_initial_vsn          = self.pooling(visual_initial_vsn)
            decoder_hidden              = [torch.cat([encoder_hidden[0].view(batch_vis,-1), visual_initial_vsn.view(batch_vis,-1)],dim=-1).unsqueeze(0),\
                                           torch.cat([encoder_hidden[1].view(batch_vis,-1), visual_initial_vsn.view(batch_vis,-1)],dim=-1).unsqueeze(0)]
        
        visual_vsn_result   = visual_initial_vsn
        
        decoder_input = obs_traj[-1,:batch_vis,:]

        a0 = encoder_hidden[0].view(batch_vis,-1)
        a1 = visual_vsn_result.view(batch_vis,-1)
        a2 = torch.einsum("bn,bm->bnm",a0,a1)
        tens_a = torch.ones(batch_vis, a0.size(1)+1, a1.size(1)+1, device="cuda")
        tens_a[:,1:,1:] = a2
        tens_a[:,0,1:]  = a1
        tens_a[:,1:,0]  = a0

        b0 = encoder_hidden[1].view(batch_vis,-1)
        b1 = visual_vsn_result.view(batch_vis,-1)
        b2 = torch.einsum("bn,bm->bnm",b0,b1)
        tens_b = torch.ones(batch_vis, b0.size(1)+1, b1.size(1)+1, device="cuda")
        tens_b[:,1:,1:] = b2
        tens_b[:,0,1:]  = b1
        tens_b[:,1:,0]  = b0

        tens_a_red = self.decoder.dim_red(tens_a[:,:17,:17])
        tens_b_red = self.decoder.dim_red(tens_b[:,:17,:17])

        # Pool States
        if self.pooling_type:
            end_pos = obs_traj[-1, :, :]
            pool_h = self.pool_net(final_encoder_h, seq_start_end, end_pos)
            # Construct input hidden states for decoder
            mlp_decoder_context_input = torch.cat(
                [final_encoder_h.view(-1, self.encoder_h_dim), pool_h], dim=1)
        else:
            mlp_decoder_context_input = final_encoder_h.view(
                -1, self.encoder_h_dim)
        
        """
        # Add Noise
        if self.mlp_decoder_needed():
            noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
        else:
            noise_input = mlp_decoder_context_input
        ##
        """

        decoder_h = [tens_a_red.unsqueeze(0)[:,:,:],tens_b_red.unsqueeze(0)[:,:,:]]

        batch_size = batch_vis
        pred_traj_fake_rel              = torch.zeros(batch_size, T_pred , in_size).cuda()#.cpu() 
        stochastic_outputs              = torch.zeros(batch_size, T_pred , stochastic_out_size).cuda()#.cpu()
        teacher_force                   = 1
        
        epsilonX                        = Normal(torch.zeros(batch_size,1),torch.ones(batch_size,1))
        epsilonY                        = Normal(torch.zeros(batch_size,1),torch.ones(batch_size,1))

        stochastic_mode         = 1
        output_tensor = output_tensor[:batch_vis,:,:]
        
        for t in range(0, T_pred):
            stochastic_decoder_output, decoder_h   = self.decoder(decoder_input, decoder_h)
            # Reparameterization Trick :)
            decoder_output              = torch.zeros(batch_vis,1,2).cuda()

            k = 5
            epsilon_x               = torch.randn([batch_size,bst_n_path_eval,1], dtype=torch.float).cuda()
            epsilon_y               = torch.randn([batch_size,bst_n_path_eval,1], dtype=torch.float).cuda()
            multi_path_x            = stochastic_decoder_output[:,:,0].unsqueeze(1) + epsilon_x * stochastic_decoder_output[:,:,1].unsqueeze(1)
            multi_path_y            = stochastic_decoder_output[:,:,2].unsqueeze(1) + epsilon_y * stochastic_decoder_output[:,:,3].unsqueeze(1)
            ground_truth_x          = output_tensor[:,t,0].view(batch_size,1,1).cuda()
            ground_truth_y          = output_tensor[:,t,1].view(batch_size,1,1).cuda()
            diff_path_x             = multi_path_x - ground_truth_x
            diff_path_y             = multi_path_y - ground_truth_y
            diff_path               = (torch.sqrt( diff_path_x.pow(2) + diff_path_y.pow(2) )).sum(dim=-1)
            idx                     = torch.arange(batch_size,dtype=torch.long).repeat(k).view(k,-1).transpose(0,1).cuda()
            min_val, min            = torch.topk(diff_path, k=k, dim=1,largest=False)
            decoder_output[:,:,0]   = multi_path_x[idx,min,:].mean(dim=-2).view(batch_size,1)
            decoder_output[:,:,1]   = multi_path_y[idx,min,:].mean(dim=-2).view(batch_size,1)

            # Log output
            pred_traj_fake_rel[:,t,:]             = decoder_output.squeeze(1)
            stochastic_outputs[:,t,:]             = stochastic_decoder_output.squeeze(1)
            decoder_input                         = output_tensor[:,t,:] if teacher_force else decoder_output

        pred_traj_fake_rel = torch.transpose(pred_traj_fake_rel,0,1)

        return pred_traj_fake_rel


class TrajectoryDiscriminator(nn.Module):
    def __init__(
        self, obs_len, pred_len, embedding_dim=64, h_dim=64, mlp_dim=1024,
        num_layers=1, activation='relu', batch_norm=True, dropout=0.0,
        d_type='local'
    ):
        super(TrajectoryDiscriminator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.d_type = d_type
        
        self.encoder = Encoder(embedding_dim=embedding_dim,h_dim=h_dim,mlp_dim=mlp_dim,num_layers=num_layers,dropout=dropout)

        real_classifier_dims = [h_dim, mlp_dim, 1]
        self.real_classifier = make_mlp(real_classifier_dims,activation=activation,batch_norm=batch_norm,dropout=dropout)
        if d_type == 'global':
            mlp_pool_dims = [h_dim + embedding_dim, mlp_dim, h_dim]
            self.pool_net = PoolHiddenNet(embedding_dim=embedding_dim,h_dim=h_dim,mlp_dim=mlp_pool_dims,bottleneck_dim=h_dim,activation=activation,batch_norm=batch_norm)

    def forward(self, traj, traj_rel, seq_start_end=None):
        """
        Inputs:
        - traj: Tensor of shape (obs_len + pred_len, batch, 2)
        - traj_rel: Tensor of shape (obs_len + pred_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - scores: Tensor of shape (batch,) with real/fake scores
        """
        final_h,_,_ = self.encoder(traj_rel)
        # Note: In case of 'global' option we are using start_pos as opposed to end_pos. The intution being that hidden state has the whole trajectory and relative postion at the start when combined with trajectory information should help in discriminative behavior.
        if self.d_type == 'local':
            classifier_input = final_h.squeeze()
        else:
            classifier_input = self.pool_net(final_h.squeeze(), seq_start_end, traj[0])
        scores = self.real_classifier(classifier_input)
        return scores
