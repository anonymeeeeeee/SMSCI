
from torch.utils.data import DataLoader
from trajectories import *

def data_loader(args, path):
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim)

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate)

    return dset, loader

def data_loader_visu(batch):
    path_visu = "data_trajpred/"+args.dataset_name
    DB_PATH_train = "data_trajpred/"+args.dataset_name+"/pos_data_train.db"
    dset_visu = TrajectoryPredictionDataset(path_visu, DB_PATH_train, sqlite3.connect(DB_PATH_train))

    loader_visu = DataLoader(
        dset_visu,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last = False,
        pin_memory = True)
    
    return loader_visu
