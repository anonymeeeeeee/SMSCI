import logging
import os
import math
from torch.utils.data import Dataset
import pandas as pd
import sqlite3
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.utils import save_image
from torch.distributions.normal import Normal
from PIL import Image
import math
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
import datetime
from skimage.util import img_as_ubyte
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import argparse

logger = logging.getLogger(__name__)

def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,non_linear_ped_list, loss_mask_list) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,loss_mask, seq_start_end]

    return tuple(out)


def read_file(_path, delim='\t'):
    data = []
    delim = 'tab'
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)

def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0

dataset_name = "eth"

__file__=os.getcwd()
print(__file__)

now = datetime.datetime.now() 
current_time_date = now.strftime("%d_%m_%y_%H_%M_%S")
run_folder  = "Outputs/traj_pred_"+ dataset_name +"_" + str(os.path.basename(__file__))+ '_' + str( datetime.datetime.now() ) 
os.makedirs(run_folder)   

# DataBase Variables
image_folder_path       = 'data_trajpred/'+dataset_name
DB_PATH_train     = "data_trajpred/"+dataset_name+"/pos_data_train.db"
cnx_train         = sqlite3.connect(DB_PATH_train)
DB_PATH_val     = "data_trajpred/"+dataset_name+"/pos_data_val.db"
cnx_val         = sqlite3.connect(DB_PATH_val)
DB_DIR      = run_folder + '/database'
os.makedirs( DB_DIR )
DB_PATH2    = DB_DIR+'/db_one_ped_delta_coordinates_results.db'
cnx2        = sqlite3.connect(DB_PATH2)

# Variables
T_obs                   = 8
T_pred                  = 12
T_total                 = T_obs + T_pred
data_id                 = 0 
batch_size              = 64
chunk_size              = batch_size * T_total
in_size                 = 2
stochastic_out_size     = in_size * 2
hidden_size             = 256 #!64
embed_size              = 64 #16 #!64
global dropout_val
dropout_val             = 0.2 #0.5
teacher_forcing_ratio   = 0.7
avg_n_path_eval         = 20
bst_n_path_eval         = 20

table       = "dataset_T_length_"+str(T_total)+"delta_coordinates"
df_id       = pd.read_sql_query("SELECT data_id FROM "+table, cnx_train)
data_size   = df_id.data_id.max() * T_total
epoch_num   = 200
from_epoch  = 0

image_size              = 256  


class TrajectoryPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, ROOT_DIR, DB_PATH,cnx):
        
        self.pos_df    = pd.read_sql_query("SELECT * FROM "+str(table), cnx)
        self.root_dir  = ROOT_DIR+'/visual_data'
        self.transform = torchvision.transforms.Compose([torchvision.transforms.Resize((image_size,image_size)), \
                                                         torchvision.transforms.ToTensor(), \
                                                         torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        self.visual_data = []
        # read sorted frames
        for img in sorted(os.listdir(self.root_dir)): 
            self.visual_data.append(self.transform( Image.open(os.path.join(self.root_dir)+"/"+img) ))
        self.visual_data = torch.stack(self.visual_data)
    
    def __len__(self):
        return self.pos_df.data_id.max()
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        extracted_df     = self.pos_df[ self.pos_df["data_id"] == idx ]
        
        tensor           = torch.tensor(extracted_df[['pos_x_delta','pos_y_delta']].values).reshape(-1,T_total,in_size)
        obs, pred        = torch.split(tensor,[T_obs,T_pred],dim=1)
        
        start_frames     = (extracted_df.groupby('data_id').frame_num.min().values/10).astype('int')
        extracted_frames = []
        for i in start_frames:            
            extracted_frames.append(self.visual_data[i:i+T_obs])
        frames = torch.stack(extracted_frames)
        start_frames = torch.tensor(start_frames)
        return obs, pred, frames, start_frames

class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002,min_ped=1, delim='\t'):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        for path in all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :]
        ]
        return out
