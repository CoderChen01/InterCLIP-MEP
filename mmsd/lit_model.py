import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Tuple, Union, cast

import pandas
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.mapping import get_peft_model
from peft.peft_model import PeftModel
from peft.tuners.lora import LoraConfig
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup

from mmsd.dataset import MMSDModelInput
from mmsd.model import InteractiveCLIP4MMSD, MMSDConfig, MMSDOutput
from mmsd.predictor import MemoEnhancedPredictor

logger = logging.getLogger("mmsd.lit_model")


class LitMMSDModel(pl.LightningModule):

    def __init__(
        self,
        clip_vision_model: str,
        clip_text_model: str,
        is_openclip: bool,
        vision_embed_dim: int,
        text_embed_dim: int,
        vision_num_layers: int,
        text_num_layers: int,
        use_sim_loss: bool = True,
        vision_cond_attn_mode: str = "top-4",
        text_cond_attn_mode: str = "top-4",
        is_v2t_adapter_mlp: bool = True,
        is_t2v_adapter_mlp: bool = False,
        memo_size: int = 512,
        use_memo: bool = True,
        embed_size: int = 1024,
        use_lora: bool = True,
        lora_modules: list[str] = ["q", "k", "v", "out"],
        lora_rank: int = 8,
        lora_alpha: int = 8,
        lora_dropout: float = 0.1,
        lora_lr: float = 1e-4,
        learning_rate: float = 5e-4,
        num_warmup_rate: float = 0.2,
        min_lr_rate: float = 0.01,
        is_compiled: bool = False,
    ) -> None:
        super().__init__()

        # model
        self.clip_vision_model = clip_vision_model
        self.clip_text_model = clip_text_model
        self.is_openclip = is_openclip
        self.vision_embed_dim = vision_embed_dim
        self.text_embed_dim = text_embed_dim
        self.vision_num_layers = vision_num_layers
        self.text_num_layers = text_num_layers
        self.embed_size = embed_size
        self.vision_cond_attn_mode = vision_cond_attn_mode
        self.text_cond_attn_mode = text_cond_attn_mode
        self.is_v2t_adapter_mlp = is_v2t_adapter_mlp
        self.is_t2v_adapter_mlp = is_t2v_adapter_mlp

        # training
        self.use_lora = use_lora
        self.lora_modules = lora_modules
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_sim_loss = use_sim_loss

        # inference
        self.memo_size = memo_size
        self.use_memo = use_memo

        # optimization
        self.is_compiled = is_compiled
        self.lora_lr = lora_lr
        self.learning_rate = learning_rate
        self.num_warmup_rate = num_warmup_rate
        self.min_lr_rate = min_lr_rate

        is_success = False
        while not is_success:
            try:
                self._init_model()
                is_success = True
            except Exception:
                logger.warning("Failed to initialize the model, retrying...")
                logger.error("Failed to initialize the model", exc_info=True)
                time.sleep(10)
                continue

        macro_metrics = MetricCollection(
            [
                Accuracy(task="multiclass", num_classes=2),
                Recall(task="multiclass", num_classes=2, average="macro"),
                F1Score(task="multiclass", num_classes=2, average="macro"),
                Precision(task="multiclass", num_classes=2, average="macro"),
            ]
        )
        binary_metrics = MetricCollection(
            [
                Accuracy(task="binary"),
                Recall(task="binary"),
                F1Score(task="binary"),
                Precision(task="binary"),
            ]
        )
        self.train_metric = binary_metrics.clone(prefix="train/")
        self.val_metric = binary_metrics.clone(prefix="val/")
        self.test_metric_macro = macro_metrics.clone(prefix="test_macro/")
        self.test_metric_binary = binary_metrics.clone(prefix="test_binary/")

        self.predictor = None
        self.case_study_results = defaultdict(list)

        self.save_hyperparameters()

    def _get_conditional_layer_ids(self, num_layers: int, mode: str) -> list[int]:
        mode_splited = mode.split("-", maxsplit=2)
        mode = mode_splited[0]

        if len(mode_splited) == 1:
            if mode not in ["all", "none"]:
                raise ValueError("Invalid conditional attention mode")
            if mode == "all":
                return [i for i in range(num_layers)]
            if mode == "none":
                return []

        if len(mode_splited) == 2:
            try:
                num = int(mode_splited[1])
            except Exception:
                raise ValueError("Invalid conditional attention mode")

            if mode == "top":
                return [i for i in range(num_layers - num, num_layers)]
            if mode == "bottom":
                return [i for i in range(num)]
            if mode == "sparse":
                return [i for i in range(0, num_layers, num)]

        raise ValueError("Invalid conditional attention mode")

    def _get_trainning_modules(
        self, model: InteractiveCLIP4MMSD
    ) -> tuple[list[str], list[str]]:
        lora_ft_modules = []
        full_training_modules = []

        for name, module in model.named_modules():
            if hasattr(module, "gating_param"):
                full_training_modules.append(f"{name}.gating_param")
                continue

            if "gated_proj" in name or (
                "adapter_proj" in name and isinstance(module, nn.Linear)
            ):
                full_training_modules.append(name)
                continue

            if ("fuse_projection" in name or "classifier" in name) and isinstance(
                module, nn.Linear
            ):
                full_training_modules.append(name)
                continue

            if not module._is_hf_initialized and isinstance(module, nn.Linear):
                full_training_modules.append(name)
                continue

            if isinstance(module, nn.Linear):
                for k in self.lora_modules:
                    if f".{k}" in name and "att" in name:
                        lora_ft_modules.append(name)
                        break

        return lora_ft_modules, full_training_modules

    def _set_full_training(
        self,
        model: Union[InteractiveCLIP4MMSD, PeftModel],
        full_training_modules: list[str],
    ) -> None:
        for name, param in model.named_parameters():
            if any([n in name for n in full_training_modules]):
                param.requires_grad = True

    def _init_model(self) -> None:
        config = MMSDConfig(
            clip_vision_model=self.clip_vision_model,
            clip_text_model=self.clip_text_model,
            is_openclip=self.is_openclip,
            vision_embed_dim=self.vision_embed_dim,
            text_embed_dim=self.text_embed_dim,
            vision_conditional_layer_ids=self._get_conditional_layer_ids(
                self.vision_num_layers, self.vision_cond_attn_mode
            ),
            text_conditional_layer_ids=self._get_conditional_layer_ids(
                self.text_num_layers, self.text_cond_attn_mode
            ),
            is_v2t_adapter_mlp=self.is_v2t_adapter_mlp,
            is_t2v_adapter_mlp=self.is_t2v_adapter_mlp,
            use_sim_loss=self.use_sim_loss,
            projection_dim=self.embed_size,
        )
        model = InteractiveCLIP4MMSD(config)

        # Set the modules that require lora fine-tuning and full training
        lora_ft_modules, full_training_modules = self._get_trainning_modules(model)
        if self.use_lora:
            model = get_peft_model(
                model,
                LoraConfig(
                    r=self.lora_rank,
                    lora_alpha=self.lora_alpha,
                    lora_dropout=self.lora_dropout,
                    target_modules=lora_ft_modules,
                ),
            )
            logger.info(f"Modules for LoRA fune-tuning: {lora_ft_modules}")
            model = cast(PeftModel, model)
        self._set_full_training(model, full_training_modules)
        logger.info(f"Modules for full training: {full_training_modules}")

        if self.is_compiled:
            self.model = torch.compile(model)
        else:
            self.model = model

    def training_step(self, batch: MMSDModelInput) -> torch.Tensor:
        output: MMSDOutput = self(**batch)
        batch_size = batch["label"].size(0)
        self.log("train/loss", output[0], prog_bar=True, batch_size=batch_size)
        self.log(
            "train/lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            prog_bar=True,
            batch_size=batch_size,
            on_epoch=False,
        )
        self.log(
            "train/lora_lr",
            self.trainer.optimizers[0].param_groups[1]["lr"],
            prog_bar=True,
            batch_size=batch_size,
            on_epoch=False,
        )
        with torch.no_grad():
            assert output.logits is not None
            pred = F.softmax(output.logits, dim=-1)
            metric_step = self.train_metric(torch.argmax(pred, dim=-1), batch["label"])
            self.log_dict(metric_step, batch_size=batch_size)
        return output[0]

    def on_train_epoch_end(self) -> None:
        metric_epoch = self.train_metric.compute()
        self.log_dict(metric_epoch)
        self.train_metric.reset()

    def validation_step(self, batch: MMSDModelInput) -> None:
        output: MMSDOutput = self(**batch)
        batch_size = batch["label"].size(0)
        assert output.logits is not None
        pred = F.softmax(output.logits, dim=-1)
        self.val_metric.update(torch.argmax(pred, dim=-1), batch["label"])
        self.log("val/loss", output[0], prog_bar=True, batch_size=batch_size)

    def on_validation_epoch_end(self) -> None:
        val_metric = self.val_metric.compute()
        self.log_dict(val_metric)
        self.val_metric.reset()

    def on_test_epoch_start(self) -> None:
        self.predictor = MemoEnhancedPredictor(
            self.model, self.use_memo, self.memo_size, self.embed_size
        )

    def test_step(self, batch: MMSDModelInput) -> None:
        if self.predictor is None:
            raise ValueError("predictor is not initialized")
        ids = cast(list[str], batch.pop("id"))
        memo_pred, pred, entropy = self.predictor(batch)

        memo_label = torch.argmax(memo_pred, dim=-1)
        cls_label = torch.argmax(pred, dim=-1)

        true_idx = memo_label == batch["label"]
        filtered_ids = [ids[i] for i in torch.nonzero(true_idx).squeeze(-1).tolist()]
        memory_true_idx = (
            torch.nonzero(cls_label[true_idx] != memo_label[true_idx])
            .squeeze(-1)
            .tolist()
        )
        cls_memory_true_idx = (
            torch.nonzero(cls_label[true_idx] == memo_label[true_idx])
            .squeeze(-1)
            .tolist()
        )
        for idx in memory_true_idx:
            self.case_study_results["id"].append(filtered_ids[idx])
            self.case_study_results["label"].append(
                batch["label"][true_idx][idx].item()
            )
            self.case_study_results["pred"].append(cls_label[true_idx][idx].item())
            self.case_study_results["memo_pred"].append(
                memo_label[true_idx][idx].item()
            )
            self.case_study_results["entropy"].append(entropy[true_idx][idx].item())
            self.case_study_results["type"].append("TrueWithOnlyMemory")
        for idx in cls_memory_true_idx:
            self.case_study_results["id"].append(filtered_ids[idx])
            self.case_study_results["label"].append(
                batch["label"][true_idx][idx].item()
            )
            self.case_study_results["pred"].append(cls_label[true_idx][idx].item())
            self.case_study_results["memo_pred"].append(
                memo_label[true_idx][idx].item()
            )
            self.case_study_results["entropy"].append(entropy[true_idx][idx].item())
            self.case_study_results["type"].append("TrueWithMemoryAndCls")

        true_idx = cls_label == batch["label"]
        filtered_ids = [ids[i] for i in torch.nonzero(true_idx).squeeze(-1).tolist()]
        cls_true_idx = (
            torch.nonzero(cls_label[true_idx] != memo_label[true_idx])
            .squeeze(-1)
            .tolist()
        )
        for idx in cls_true_idx:
            self.case_study_results["id"].append(filtered_ids[idx])
            self.case_study_results["label"].append(
                batch["label"][true_idx][idx].item()
            )
            self.case_study_results["pred"].append(cls_label[true_idx][idx].item())
            self.case_study_results["memo_pred"].append(
                memo_label[true_idx][idx].item()
            )
            self.case_study_results["entropy"].append(entropy[true_idx][idx].item())
            self.case_study_results["type"].append("TrueWithOnlyCls")

        false_idx = memo_label != batch["label"]
        filtered_ids = [ids[i] for i in torch.nonzero(false_idx).squeeze(-1).tolist()]
        cls_memo_false_idx = (
            torch.nonzero(cls_label[false_idx] == memo_label[false_idx])
            .squeeze(-1)
            .tolist()
        )
        for idx in cls_memo_false_idx:
            self.case_study_results["id"].append(filtered_ids[idx])
            self.case_study_results["label"].append(
                batch["label"][false_idx][idx].item()
            )
            self.case_study_results["pred"].append(cls_label[false_idx][idx].item())
            self.case_study_results["memo_pred"].append(
                memo_label[false_idx][idx].item()
            )
            self.case_study_results["entropy"].append(entropy[false_idx][idx].item())
            self.case_study_results["type"].append("FalseWithMemoryAndCls")

        self.test_metric_macro.update(memo_pred, batch["label"])
        self.test_metric_binary.update(memo_label, batch["label"])

    def on_test_epoch_end(self) -> None:
        test_metric_macro = self.test_metric_macro.compute()
        self.log_dict(test_metric_macro)
        test_metric_micro = self.test_metric_binary.compute()
        self.log_dict(test_metric_micro)
        self.test_metric_macro.reset()
        self.test_metric_binary.reset()
        del self.predictor
        self.predictor = None

        assert self.trainer.log_dir is not None and self.trainer.ckpt_path is not None
        case_study_path = (
            Path(self.trainer.log_dir)
            / f"case_study_{Path(self.trainer.ckpt_path).name}.csv"
        )
        pandas.DataFrame(self.case_study_results).to_csv(case_study_path, index=False)
        self.case_study_results.clear()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        lora_params = []
        params = []

        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "lora" in n:
                lora_params.append(p)
            else:
                params.append(p)

        optimizer = torch.optim.AdamW(
            [
                {"params": params, "lr": self.learning_rate},
                {"params": lora_params, "lr": self.lora_lr},
            ]
        )
        lr_scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(
                self.num_warmup_rate * self.trainer.estimated_stepping_batches
            ),
            num_training_steps=int(self.trainer.estimated_stepping_batches),
            min_lr_rate=self.min_lr_rate,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        label: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, MMSDOutput]:
        return self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            label=label,
            return_loss=return_loss,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
