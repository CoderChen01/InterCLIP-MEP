import json
import shutil
import typing as t
from pathlib import Path

import datasets as dt
import jsonlines
from tqdm import tqdm

VERSION_TYPE = t.Literal["mmsd-v1", "mmsd-v2", "mmsd-original"]

MMSD2_DATASET_DIR = Path(
    "/home/chenjunjie/projects/paper/sarcasm-ws/methods/MMSD2.0/data"
)

VERSION_NAME_MAP = {
    "mmsd-v1": "text_json_clean",
    "mmsd-v2": "text_json_final",
    "mmsd-original": "original",
}


def read_orignal(split_path: Path) -> list[dict[str, t.Any]]:
    lines = split_path.read_text().splitlines()

    all_data = []
    for line in lines:
        data = eval(line)
        text = data[1].split()

        if "sarcasm" in text:
            continue
        if "sarcastic" in text:
            continue
        if "reposting" in text:
            continue
        if "<url>" in text:
            continue
        if "joke" in text:
            continue
        if "humour" in text:
            continue
        if "humor" in text:
            continue
        if "jokes" in text:
            continue
        if "irony" in text:
            continue
        if "ironic" in text:
            continue
        if "exgag" in text:
            continue

        all_data.append({"image_id": data[0], "text": data[1], "label": int(data[-1])})

    return all_data


def convert(version: VERSION_TYPE) -> None:
    data_dir = MMSD2_DATASET_DIR / VERSION_NAME_MAP[version]

    # create converted data dir
    converted_data_dir = MMSD2_DATASET_DIR / f"{version}-converted"
    converted_data_dir.mkdir(exist_ok=True, parents=True)

    for split in ["test", "valid", "train"]:
        # create split dir
        split_dir = converted_data_dir / split
        split_dir.mkdir(exist_ok=True, parents=True)

        # create metadata file
        metadata_file = (split_dir / "metadata.jsonl").open("w")
        metadata_writer = jsonlines.Writer(metadata_file)

        # copy images and write metadata
        if version == "mmsd-original":
            split = f"{split}2" if split != "train" else split
            data = read_orignal(data_dir / f"{split}.txt")
        else:
            data = json.loads((data_dir / f"{split}.json").read_text())

        for d in tqdm(data, desc=f"Converting {version} {split} data"):
            image_id = d["image_id"]
            text = d["text"]
            label = d["label"]
            image_path = MMSD2_DATASET_DIR / "images" / f"{image_id}.jpg"
            if not image_path.exists():
                continue
            metadata_writer.write(
                {
                    "file_name": f"{image_id}.jpg",
                    "text": text,
                    "label": label,
                    "id": str(image_id),
                }
            )
            shutil.copy(image_path, split_dir / f"{image_id}.jpg")


def publish(version: VERSION_TYPE, repo_id: str) -> None:
    converted_data_dir = MMSD2_DATASET_DIR / f"{version}-converted"
    dataset = t.cast(
        dt.Dataset, dt.load_dataset("imagefolder", data_dir=str(converted_data_dir))
    )
    dataset.push_to_hub(repo_id, config_name=version)


if __name__ == "__main__":
    convert("mmsd-v1")
    convert("mmsd-v2")
    convert("mmsd-original")

    publish("mmsd-v1", "coderchen01/MMSD2.0")
    publish("mmsd-v2", "coderchen01/MMSD2.0")
    publish("mmsd-original", "coderchen01/MMSD2.0")
