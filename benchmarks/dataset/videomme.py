# SPDX-License-Identifier: Apache-2.0
"""Video-MME dataset loader for local benchmarks."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


@dataclass
class VideoMMESample:
    sample_id: str
    video_path: str
    question: str
    options: list[str]
    answer: str
    url: str = ""
    video_id: str = ""
    question_id: str = ""
    duration: str = "short"
    domain: str = "unknown"
    task_type: str = "understanding"
    sub_category: str = ""
    prompt: str = ""
    all_choices: list[str] = field(default_factory=list)
    index2ans: dict[str, str] = field(default_factory=dict)


def _strip_option_prefix(option: str) -> str:
    return re.sub(r"^[A-D]\.\s*", "", option.strip())


def format_videomme_prompt(question: str, options: list[str]) -> str:
    prompt = f"{question.strip()}\n"
    for index, option in enumerate(options):
        letter = chr(ord("A") + index)
        prompt += f"{letter}. {option}\n"
    prompt += (
        "\nAnswer the following multiple-choice question. "
        "The last line of your response should be of the "
        "following format: 'Answer: $LETTER' (without quotes) "
        "where LETTER is one of the options. "
        "Think step by step before answering."
    )
    return prompt


def _resolve_video_path(snapshot_dir: Path, row: dict, question_id: str) -> str | None:
    relative_path = row.get("video_path")
    if not relative_path:
        logger.warning(
            "Skipping Video-MME sample %s because the dataset row has no video_path",
            question_id,
        )
        return None
    absolute_path = (snapshot_dir / str(relative_path)).resolve()
    if not absolute_path.exists():
        logger.warning(
            "Skipping Video-MME sample %s because the video file does not exist at %s",
            question_id,
            absolute_path,
        )
        return None
    return str(absolute_path)


def _dataset_to_samples(
    dataset,
    *,
    snapshot_dir: Path,
    max_samples: int | None,
) -> list[VideoMMESample]:
    samples: list[VideoMMESample] = []
    for row_index, row in enumerate(dataset):
        duration = str(row.get("duration", "short")).strip()
        question_id = str(row.get("question_id", f"videomme:{row_index}")).strip()

        options = [_strip_option_prefix(str(option)) for option in row["options"]]
        all_choices = [chr(ord("A") + i) for i in range(len(options))]
        index2ans = {choice: option for choice, option in zip(all_choices, options)}
        video_id = str(row["video_id"]).strip()
        url = str(row["url"]).strip()
        video_path = _resolve_video_path(snapshot_dir, row, question_id)
        if not video_path:
            continue

        samples.append(
            VideoMMESample(
                sample_id=question_id,
                video_path=video_path,
                question=str(row["question"]).strip(),
                options=options,
                answer=str(row["answer"]).strip(),
                url=url,
                video_id=video_id,
                question_id=question_id,
                duration=duration,
                domain=str(row.get("domain", "unknown")).strip(),
                task_type=str(row.get("task_type", "understanding")).strip(),
                sub_category=str(row.get("sub_category", "")).strip(),
                prompt=format_videomme_prompt(str(row["question"]).strip(), options),
                all_choices=all_choices,
                index2ans=index2ans,
            )
        )
        if max_samples is not None and len(samples) >= max_samples:
            break

    return samples


def _load_metadata_dataset(snapshot_dir: Path, split: str):
    data_dir = snapshot_dir / "data"
    split_parts = sorted(data_dir.glob(f"{split}_part_*.jsonl"))
    if split_parts:
        return load_dataset(
            "json",
            data_files=[str(path) for path in split_parts],
            split="train",
        )

    split_file = data_dir / f"{split}.jsonl"
    if split_file.exists():
        return load_dataset("json", data_files=str(split_file), split="train")

    available = sorted(path.name for path in data_dir.glob("*.jsonl"))
    raise ValueError(
        f"Split '{split}' not found under {data_dir}. Available files: {available}"
    )


def load_videomme_samples(
    max_samples: int | None = None,
    *,
    repo_id: str | None = None,
    split: str = "test",
) -> list[VideoMMESample]:
    resolved_repo_id = repo_id or "zhaochenyang20/Video_MME"
    snapshot_dir = Path(
        snapshot_download(repo_id=resolved_repo_id, repo_type="dataset")
    )
    dataset = _load_metadata_dataset(snapshot_dir, split)
    samples = _dataset_to_samples(
        dataset,
        snapshot_dir=snapshot_dir,
        max_samples=max_samples,
    )
    logger.info("Loaded %d Video-MME samples", len(samples))
    return samples
