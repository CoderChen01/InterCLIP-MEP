import typing as t

import datasets as dt

from mmsd.dataset import MMSDDatasetModule


def test_dataset_module() -> None:
    dm = MMSDDatasetModule(
        "openai/clip-vit-base-patch32", "openai/clip-vit-base-patch32"
    )
    dm.setup("fit")
    d = dm.train_dataloader()
    sample = d.__iter__().__next__()
    assert sample["pixel_values"].shape[0] == 32
