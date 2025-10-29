import os

from torch.utils.data import DataLoader

from Occ.Occ_args import init_sub_args
from Occ.models.STG_NF.model_pose import STG_NF
from Occ.models.training import Trainer
from Occ.utils.optim_init import init_optimizer, init_scheduler
from Occ.utils.scoring_utils import score_dataset
from Occ.utils.train_utils import init_model_params
from Occ.utils.training_scoring_utils import training_score_dataset


def prepare_occ_trainer(Occ_args=None, dataset=None, pseudo_weight=None, dir_best_record="", load=False):
    args, model_args = init_sub_args(Occ_args)
    checkpoint_path = os.path.join(dir_best_record, "Occ_epoch_final_checkpoint.pth.tar") if load else None

    if pseudo_weight is not None:
        dataset["train"].update_weights(pseudo_weight)

    loader_args = {"batch_size": args.Occ_batch_size, "num_workers": args.Occ_num_workers, "pin_memory": True}
    loader_train = DataLoader(dataset["train"], **loader_args, shuffle=True)
    loader_train_eval = DataLoader(dataset["train_test"], **loader_args, shuffle=False)
    loader_test = DataLoader(dataset["test"], **loader_args, shuffle=False)

    model_args = init_model_params(args, dataset)
    model = STG_NF(**model_args)
    trainer = Trainer(
        args,
        model,
        loader_train,
        loader_train_eval,
        loader_test,
        optimizer_f=init_optimizer(args.Occ_model_optimizer, lr=args.Occ_model_lr),
        scheduler_f=init_scheduler(args.model_sched, lr=args.Occ_model_lr, epochs=args.Occ_epochs),
    )
    if checkpoint_path:
        trainer.load_checkpoint(checkpoint_path, args)
    return args, trainer, checkpoint_path


def Occ(
    Occ_args=None,
    dataset=None,
    pseudo_weight=None,
    dir_best_record="",
    load=False,
    writer=None,
    steps=0,
):
    args, trainer, _ = prepare_occ_trainer(
        Occ_args=Occ_args,
        dataset=dataset,
        pseudo_weight=pseudo_weight,
        dir_best_record=dir_best_record,
        load=load,
    )

    trainer.train(log_writer=writer, dir_best_record=dir_best_record, steps=steps)

    normality_scores, nll_values = trainer.training_scores_producer()
    training_scores = training_score_dataset(normality_scores, dataset["train_test"].metadata, args=args)

    test_scores = trainer.test()
    test_auc, roc_scores, roc_gt = score_dataset(test_scores, dataset["test"].metadata, args=args)

    print("\n-------------------------------------------------------")
    print("\033[92m Done with {}% AuC for {} samples\033[0m".format(test_auc * 100, training_scores.shape[0]))
    print("-------------------------------------------------------\n\n")

    details = {
        "nll_values": nll_values,
        "roc_scores": roc_scores,
        "roc_gt": roc_gt,
        "checkpoint_path": os.path.join(dir_best_record, "Occ_epoch_final_checkpoint.pth.tar"),
    }
    return training_scores, test_auc, details
