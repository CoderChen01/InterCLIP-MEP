from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
import yaml
from pytorch_lightning import Trainer

from mmsd.dataset import MMSDDatasetModule
from mmsd.lit_model import LitMMSDModel
from mmsd.predictor import MemoEnhancedPredictor


def get_model_and_data(path: str) -> tuple[MMSDDatasetModule, LitMMSDModel]:
    path_obj = Path(path)
    args = yaml.load(path_obj.open("r"), Loader=yaml.FullLoader)
    dataset_module = MMSDDatasetModule(**args["data"])
    model = LitMMSDModel(**args["model"]["init_args"])
    return dataset_module, model


def get_experiments(path: str) -> list[Path]:
    path_obj = Path(path)
    experiments = path_obj.glob("clip-*")
    return list(experiments)


def get_ckpt_paths(experiment: Path) -> list[Path]:
    path_obj = experiment
    ckpt_path = path_obj / "checkpoints"
    ckpts = list(ckpt_path.glob("*.ckpt"))
    epoch_ckpts: dict[str, Path] = {}
    for ckpt in ckpts:
        ckpt_name = ckpt.name
        epoch = ckpt_name.split("-")[1].split("=")[1]
        epoch_ckpts[epoch] = ckpt
    return list(epoch_ckpts.values())


def grid_search_memo_size(
    main_path: str,
    dataset_module: MMSDDatasetModule,
    memo_sizes: list[int],
    trainer: Trainer,
) -> dict:
    experiments = get_experiments(main_path)
    experiment_results = {}
    print("Start to test the model.")
    for experiment in experiments:
        exp_name = experiment.name
        ckpt_paths = get_ckpt_paths(experiment)
        results = {}
        for ckpt_path in ckpt_paths:
            ckpt_name = ckpt_path.name
            memo_results = {}
            for memo_size in memo_sizes:
                is_success = False
                while not is_success:
                    try:
                        model = LitMMSDModel.load_from_checkpoint(
                            ckpt_path, memo_size=memo_size
                        )
                        print(
                            "Run experiment:",
                            exp_name,
                            "ckpt:",
                            ckpt_name,
                            "memo_size:",
                            memo_size,
                        )
                        rv = trainer.test(
                            model, datamodule=dataset_module, verbose=False
                        )[0]
                        is_success = True
                    except Exception as e:
                        pass
                memo_results[memo_size] = rv
            results[ckpt_name] = memo_results
        experiment_results[exp_name] = results
        print("Finish experiment:", exp_name)
    print("Finish all experiments.")
    return experiment_results


def format_experiment_results(experiment_results: dict, save_path: str) -> None:
    all_results = defaultdict(list)
    stat_results = defaultdict(list)
    print("Start to save results.")
    for exp_name in experiment_results:
        exp_results = experiment_results[exp_name]
        for ckpt_name in exp_results:
            ckpt_res = exp_results[ckpt_name]
            # Save all results
            for size, rv in ckpt_res.items():
                all_results["exp_name"].append(exp_name)
                all_results["ckpt_name"].append(ckpt_name)
                all_results["memo_size"].append(size)
                all_results["Acc."].append(rv["test_binary/BinaryAccuracy"])
                all_results["Precision"].append(rv["test_binary/BinaryPrecision"])
                all_results["Recall"].append(rv["test_binary/BinaryRecall"])
                all_results["F1"].append(rv["test_binary/BinaryF1Score"])
            # Save optimal results
            memo_sizes = list(ckpt_res.keys())
            metrics = defaultdict(list)
            memo_res = list(ckpt_res.values())
            for r in memo_res:
                metrics["f1"].append(r["test_binary/BinaryF1Score"])
                metrics["acc"].append(r["test_binary/BinaryAccuracy"])
            max_f1_idx = metrics["f1"].index(max(metrics["f1"]))
            max_acc_idx = metrics["acc"].index(max(metrics["acc"]))
            max_f1_size = memo_sizes[max_f1_idx]
            max_acc_size = memo_sizes[max_acc_idx]
            stat_results["exp_name"].append(exp_name)
            stat_results["ckpt_name"].append(ckpt_name)
            stat_results["max_f1_memo_size"].append(max_f1_size)
            stat_results["max_f1"].append(metrics["f1"][max_f1_idx])
            stat_results["max_acc_memo_size"].append(max_acc_size)
            stat_results["max_acc"].append(metrics["acc"][max_acc_idx])

    save_dir = Path(save_path)
    save_dir.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(all_results).to_csv(str(save_dir / "all_results.csv"))
    pd.DataFrame(stat_results).to_csv(str(save_dir / "stat_results.csv"))
    print("Finish saving results.")


if __name__ == "__main__":
    memo_sizes = [128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280]
    main_path = "/data/chenjunjie/mmsd-results/tb_logs/lightning_logs"
    dataset_module = MMSDDatasetModule(
        vision_ckpt_name="openai/clip-vit-base-patch32",
        text_ckpt_name="openai/clip-vit-base-patch32",
        dataset_version="mmsd-v2",
        train_batch_size=64,
        val_batch_size=128,
        test_batch_size=128,
    )
    torch.set_float32_matmul_precision("medium")
    trainer = Trainer(precision="16-mixed", logger=False)
    results = grid_search_memo_size(main_path, dataset_module, memo_sizes, trainer)
    save_path = "/data/chenjunjie/mmsd-results/memo_exps/"
    format_experiment_results(results, save_path)
