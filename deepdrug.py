import os
import warnings
import numpy as np
import yaml
import pandas as pd
warnings.filterwarnings('ignore')
import argparse
import torch as t
from torch import Tensor
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning import Trainer

from model import DeepDrug_Container
from dataset import DeepDrug_Dataset
from utils import * 

def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('-cf', '--configfile', type=str, default='')
    return parser


if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()
    print(args)

    configfile = args.configfile
    with open(configfile, 'r') as f:
        # dùng safe_load để tương thích PyYAML mới
        config = yaml.safe_load(f)
        print(config)
    args = Struct(**config)

    entry1_data_folder = '/'.join(args.entry1_file.split('/')[:-2])
    entry2_data_folder = '/'.join(args.entry2_file.split('/')[:-2])
    entry2_seq_file = args.entry2_seq_file
    entry1_seq_file = args.entry1_seq_file
    assert os.path.exists(entry1_seq_file), 'file does not exist: %s.' % entry1_seq_file
    assert os.path.exists(entry2_seq_file), 'file does not exist: %s.' % entry2_seq_file
    entry_pairs_file = args.pair_file
    pair_labels_file = args.label_file
    save_folder = args.save_folder
    dataset = args.dataset
    save_model_folder = pathjoin(save_folder, 'models')
    y_true_file = pathjoin(save_folder, 'test_true.csv')
    y_pred_file = pathjoin(save_folder, 'test_pred.csv')
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(pathjoin(save_folder, 'plots'), exist_ok=True)
    task_type = args.task
    dataset = args.dataset
    gpus = args.gpus
    category = args.category
    num_out_dim = args.num_out_dim

    y_transfrom_func = None
    if (dataset in ['DAVIS', 'BindingDB']) and (task_type == 'regression'):
        y_transfrom_func = y_log10_transfrom_func

    if args.task in ['binary', 'multiclass', 'multilabel']:
        scheduler_ReduceLROnPlateau_tracking = 'F1'
        earlystopping_tracking = "val_epoch_F1"
    else:
        earlystopping_tracking = 'val_loss'
        scheduler_ReduceLROnPlateau_tracking = 'mse'

    kwargs_dict = dict(
        save_folder=save_folder,
        task_type=task_type,
        gpus=gpus,
        entry1_data_folder=entry1_data_folder,
        entry2_data_folder=entry2_data_folder, entry_pairs_file=entry_pairs_file,
        pair_labels_file=pair_labels_file,
        entry1_seq_file=entry1_seq_file, entry2_seq_file=entry2_seq_file,
        y_true_file=y_true_file, y_pred_file=y_pred_file,
        y_transfrom_func=y_transfrom_func,
        earlystopping_tracking=earlystopping_tracking,
        scheduler_ReduceLROnPlateau_tracking=scheduler_ReduceLROnPlateau_tracking,
    )

    _ = print_args(**kwargs_dict)

    datamodule = DeepDrug_Dataset(
        entry1_data_folder, entry2_data_folder, entry_pairs_file,
        pair_labels_file,
        task_type=task_type,
        y_transfrom_func=y_transfrom_func,
        entry2_seq_file=entry2_seq_file,
        entry1_seq_file=entry1_seq_file,
        category=category,
    )

    model = DeepDrug_Container(
        task_type=task_type, category=category,
        scheduler_ReduceLROnPlateau_tracking=scheduler_ReduceLROnPlateau_tracking,
        num_out_dim=num_out_dim,
    )

    # Thiết lập EarlyStopping/Checkpoint theo Lightning 2.x
    if earlystopping_tracking in ['val_loss']:
        earlystopping_mode = 'min'
        earlystopping_min_delta = 1e-4
    elif earlystopping_tracking in ['val_epoch_F1', 'val_epoch_auPRC']:
        earlystopping_mode = 'max'
        earlystopping_min_delta = 1e-3
    else:
        raise ValueError(f"Unknown monitor: {earlystopping_tracking}")

    checkpoint_callback = pl_callbacks.ModelCheckpoint(
        dirpath=save_model_folder,
        filename='{epoch}-{'+earlystopping_tracking+':.4f}',
        monitor=earlystopping_tracking,
        mode=earlystopping_mode,
        save_top_k=1,
        save_last=True,
    )

    earlystop_callback = pl_callbacks.EarlyStopping(
        monitor=earlystopping_tracking,
        verbose=True,
        mode=earlystopping_mode,
        min_delta=earlystopping_min_delta,
        patience=10,
    )

    # Lightning 2.x: dùng accelerator/devices thay cho gpus
    use_gpu = t.cuda.is_available()
    trainer = Trainer(
        accelerator="gpu" if use_gpu else "cpu",
        devices=1 if use_gpu else "auto",
        max_epochs=200,
        min_epochs=5,
        default_root_dir=save_folder,
        fast_dev_run=False,
        check_val_every_n_epoch=1,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback, earlystop_callback],
        # logger: giữ mặc định của Lightning; thêm logger nếu bạn có
    )

    trainer.fit(model, datamodule=datamodule)

    ################  Prediction ##################
    print(f'loading best weight in {checkpoint_callback.best_model_path} ...')
    # Lightning 2.x: load_from_checkpoint là @classmethod
    model = model.__class__.load_from_checkpoint(checkpoint_callback.best_model_path)
    model.eval()

    trainer.test(model, datamodule=datamodule)

    y_pred_batches = trainer.predict(model, dataloaders=datamodule.test_dataloader())
    y_true = np.array(datamodule.pair_labels[datamodule.test_indexs])

    # gộp batch predict
    def to_np(x):
        if isinstance(x, t.Tensor):
            return x.detach().cpu().numpy()
        return x

    y_pred_batches = [to_np(batch) for batch in y_pred_batches]
    y_pred = np.concatenate(y_pred_batches, axis=0)

    pd.DataFrame(y_pred).to_csv(y_pred_file, header=True, index=False)
    pd.DataFrame(y_true).to_csv(y_true_file, header=True, index=False)
    print('save prediction completed.')
