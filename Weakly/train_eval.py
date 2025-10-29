import os
from typing import Dict, Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from Weakly.config import Config
from Weakly.dataset import Dataset
from Weakly.datasets.produce_pool import pool_creation
from Weakly.model import Model
from Weakly.test_10crop import test
from Weakly.train import train

# viz = Visualizer(env='shanghai tech 10 crop', use_incoming_socket=False)

def prepare_weakly_environment(args, pseudo_idx=None) -> Tuple[Config, torch.device, Model, Dict[str, DataLoader]]:
    config = Config(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    normal_bag_list, abnormal_bag_list = pool_creation(
        data_path=args.rgb_list,
        bag_size=args.bag_size,
        test_mode=False,
        pseudo_idx=pseudo_idx,
        dataset=args.dataset,
        extractor=args.feat_extractor,
    )

    loaders: Dict[str, DataLoader] = {}
    loaders["train_normal"] = DataLoader(
        Dataset(args, test_mode=False, is_normal=True, bag_list=normal_bag_list),
        batch_size=args.Weakly_batch_size,
        shuffle=True,
        num_workers=args.Weakly_workers,
        pin_memory=False,
        drop_last=True,
    )
    loaders["train_abnormal"] = DataLoader(
        Dataset(args, test_mode=False, is_normal=False, bag_list=abnormal_bag_list),
        batch_size=args.Weakly_batch_size,
        shuffle=True,
        num_workers=args.Weakly_workers,
        pin_memory=False,
        drop_last=True,
    )
    loaders["test"] = DataLoader(
        Dataset(args, test_mode=True, bag_list=None),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    model = Model(args.feature_size, args.Weakly_batch_size, args.feat_extractor)
    return config, device, model, loaders


def Weakly(
    Weakly_args=None,
    pseudo_idx=None,
    dir_best_record="",
    load=False,
    writer=None,
    steps=0,
):
    args = Weakly_args
    config, device, model, loaders = prepare_weakly_environment(args, pseudo_idx=pseudo_idx)

    if load:
        print("loading pretrained_ckpt")
        pre_ckpt = torch.load(os.path.join(dir_best_record, "Weakly_epoch_final_checkpoint.pkl"))
        model.load_state_dict(pre_ckpt)

    model = model.to(device)

    if not os.path.exists("checkpoints/"):
        os.makedirs("checkpoints/")

    optimizer = optim.Adam(model.parameters(), lr=config.lr[0], weight_decay=0.005)

    test_info = {"epoch": [], "test_AUC": []}
    best_auc = -1
    cur_best_path = os.path.join(dir_best_record, "Weakly_epoch_final_checkpoint.pkl")
    inner_best_path = os.path.join(dir_best_record, "Weakly_inner_best_checkpoint.pkl")

    train_nloader = loaders["train_normal"]
    train_aloader = loaders["train_abnormal"]
    test_loader = loaders["test"]

    for step in tqdm(
        range(1, args.Weakly_max_epoch + 1),
        total=args.Weakly_max_epoch,
        dynamic_ncols=True,
    ):
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]

        if (step - 1) % len(train_nloader) == 0:
            loadern_iter = iter(train_nloader)

        if (step - 1) % len(train_aloader) == 0:
            loadera_iter = iter(train_aloader)

        train(
            loadern_iter,
            loadera_iter,
            model,
            args.Weakly_batch_size,
            optimizer,
            device,
            writer=writer,
            step=step,
            rounds=steps * args.Weakly_max_epoch,
        )

        if step % 5 == 0 and step >= 200:
            auc, _, _ = test(test_loader, model, args, device, return_curve=True)
            test_info["epoch"].append(step)
            test_info["test_AUC"].append(auc)

            if test_info["test_AUC"][-1] > best_auc:
                best_auc = test_info["test_AUC"][-1]
                torch.save(model.state_dict(), inner_best_path)

    torch.save(model.state_dict(), cur_best_path)
    # Evaluate final model on test set
    final_auc, fpr, tpr = test(test_loader, model, args, device, return_curve=True)
    # Scores for Occ (training set predictions)
    pred = Weakly_update(cur_best_path, args, device, update=True)

    details = {
        "checkpoint_path": cur_best_path,
        "best_checkpoint_path": inner_best_path if os.path.exists(inner_best_path) else cur_best_path,
        "auc_curve": {"auc": final_auc, "fpr": fpr, "tpr": tpr},
        "train_scores": pred,
        "test_history": test_info,
        "device": str(device),
    }
    return pred, best_auc, details


def Weakly_update(checkpoint_path, args, device, update=True, state_dict=None):
    model = Model(args.feature_size, args.Weakly_batch_size, args.feat_extractor)
    model = model.to(device)
    if state_dict is None:
        loaded_state_dict = torch.load(checkpoint_path)
    else:
        loaded_state_dict = state_dict
    model.load_state_dict(loaded_state_dict)
    test_loader = DataLoader(
        Dataset(args, test_mode=True, bag_list=None, update=update),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    pred = test(test_loader, model, args, device, update=True)
    return pred
