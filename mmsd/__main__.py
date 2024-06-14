import logging

logger = logging.getLogger("mmsd")
logger.propagate = False
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

from pathlib import Path
from typing import cast

import numpy as np
import torch
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.cli import LightningCLI

from mmsd.dataset import MMSDDatasetModule
from mmsd.lit_model import LitMMSDModel


class MMSDCLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.add_argument("--result-save-path", type=str, default=None)
        parser.add_argument("--run-test", action="store_true")
        parser.add_argument("--search-memo-size", action="store_true")
        parser.add_argument("--memo-size-save-path", type=str, default=None)

    def after_fit(self) -> None:
        if not self.config.fit["run_test"]:
            return

        rv_file = None
        if self.config.fit["result_save_path"] is not None:
            result_save_path_obj = Path(self.config.fit["result_save_path"])
            if result_save_path_obj.exists() and result_save_path_obj.is_file():
                rv_file = result_save_path_obj.open("a")
            else:
                rv_file = result_save_path_obj.open("w")

        memo_rv_file = None
        if (
            self.config.fit["search_memo_size"]
            and self.config.fit["memo_size_save_path"] is not None
        ):
            memo_size_save_path_obj = Path(self.config.fit["memo_size_save_path"])
            if memo_size_save_path_obj.exists() and memo_size_save_path_obj.is_file():
                memo_rv_file = memo_size_save_path_obj.open("a")
            else:
                memo_rv_file = memo_size_save_path_obj.open("w")

        logger.info("Start to test the model.")
        visited_epoch = []
        max_acc = -np.inf
        max_rv = None
        max_ckpt = None
        max_memo_size = None
        for ckpt in self.trainer.checkpoint_callbacks:
            ckpt = cast(ModelCheckpoint, ckpt)
            p = Path(ckpt.best_model_path)
            epoch = p.name.split("-")[1].split("=")[1]
            if epoch in visited_epoch:
                continue
            visited_epoch.append(epoch)

            logger.info(
                f"Testing the model with the best checkpoint: {ckpt.best_model_path}, the best metric: {ckpt.best_model_score}"
            )

            if self.config.fit["search_memo_size"]:
                memo_sizes = [128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280]
                for memo_size in memo_sizes:
                    model = LitMMSDModel.load_from_checkpoint(p, memo_size=memo_size)
                    if (
                        self.config.fit["result_save_path"] is None
                        and memo_rv_file is None
                    ):
                        self.trainer.test(
                            model,
                            datamodule=self.datamodule,
                            ckpt_path=ckpt.best_model_path,
                        )
                        continue
                    result = self.trainer.test(
                        model,
                        datamodule=self.datamodule,
                        ckpt_path=ckpt.best_model_path,
                        verbose=False,
                    )
                    if memo_rv_file is not None:
                        rv = result[0]
                        acc = round(rv["test_binary/BinaryAccuracy"] * 100, 2)
                        f1 = round(rv["test_binary/BinaryF1Score"] * 100, 2)
                        pr = round(rv["test_binary/BinaryPrecision"] * 100, 2)
                        r = round(rv["test_binary/BinaryRecall"] * 100, 2)
                        cmd = "#".join(self.parser.args)
                        version_name = p.parent.parent.name
                        ckpt_name = p.name
                        memo_rv_file.write(
                            f"{cmd}    {version_name}/{ckpt_name}    {memo_size}    {acc}    {f1}    {pr}    {r}\n"
                        )
                    acc = result[0]["test_binary/BinaryAccuracy"]
                    if acc <= max_acc:
                        continue
                    max_acc = acc
                    max_rv = result[0]
                    max_ckpt = ckpt.best_model_path
                    max_memo_size = memo_size

            elif self.config.fit["result_save_path"] is None:
                self.trainer.test(
                    self.model,
                    datamodule=self.datamodule,
                    ckpt_path=ckpt.best_model_path,
                )

            else:
                result = self.trainer.test(
                    self.model,
                    datamodule=self.datamodule,
                    ckpt_path=ckpt.best_model_path,
                    verbose=False,
                )
                if result[0]["test_binary/BinaryAccuracy"] <= max_acc:
                    continue
                max_acc = result[0]["test_binary/BinaryAccuracy"]
                max_rv = result[0]
                max_ckpt = ckpt.best_model_path

        if rv_file is not None:
            assert isinstance(max_ckpt, str)
            assert max_rv is not None

            acc = round(max_rv["test_binary/BinaryAccuracy"] * 100, 2)
            f1 = round(max_rv["test_binary/BinaryF1Score"] * 100, 2)
            p = round(max_rv["test_binary/BinaryPrecision"] * 100, 2)
            r = round(max_rv["test_binary/BinaryRecall"] * 100, 2)

            version_name = Path(max_ckpt).parent.parent.name
            ckpt_name = Path(max_ckpt).name

            cmd = "#".join(self.parser.args)
            if max_memo_size is not None:
                cmd += f"(memo_size={max_memo_size})"
            rv_file.write(
                f'"{cmd}"    {version_name}/{ckpt_name}    {acc}    {f1}    {p}    {r}\n'
            )
            rv_file.close()


def cli_main() -> None:
    torch.set_float32_matmul_precision("medium")
    MMSDCLI(model_class=LitMMSDModel, datamodule_class=MMSDDatasetModule)


if __name__ == "__main__":
    cli_main()
