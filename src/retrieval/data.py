"""Load InsuranceQA V2 behind a data contract.

The files are downloaded at build time from a pinned commit and checked against
known hashes, because every published number depends on these exact bytes. The
dataset is released for research use only, so it is never committed here.

Feng et al., "Applying Deep Learning to Answer Selection: A Study and an Open
Task", ASRU 2015. https://github.com/shuzi/insuranceQA
"""
from __future__ import annotations

import gzip
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "insuranceqa"

COMMIT = "5c380dd086067adacc2fec3bfa920a78e57bdcf6"
BASE_URL = f"https://raw.githubusercontent.com/shuzi/insuranceQA/{COMMIT}/V2/"
VOCABULARY = "vocabulary"
ANSWERS = "InsuranceQA.label2answer.raw.encoded.gz"
SPLIT_FILE = "InsuranceQA.question.anslabel.raw.100.pool.solr.{split}.encoded.gz"
SPLITS = ("train", "valid", "test")

FILES = {
    VOCABULARY: "087ad8bd0254b57711082856b24d3526178b407d6f849798af1e34f599b2557d",
    ANSWERS: "25f63201c51e3e2e5c73ca30da114bf8c77c8f40194550148e469a430cceabd2",
    SPLIT_FILE.format(split="train"):
        "7da9e607180ad5c47cb4069c7480cb4eb198ef88f59a2be62383a67ccb2a6981",
    SPLIT_FILE.format(split="valid"):
        "4d594fb3aec673ab61156bacc3adbea8e189215620acd284638b0d0ce0ad55e7",
    SPLIT_FILE.format(split="test"):
        "7485b38ed538bf9ce9f416efdaa9b13a699ac428dc1dc23fbcae1e57ce398d90",
}

# Published sizes. A mismatch means the files are not what the numbers describe.
EXPECTED = {"answers": 27_413, "train": 12_889, "valid": 2_000, "test": 2_000}


class DataContractError(RuntimeError):
    """The data does not match what every downstream number assumes."""


@dataclass(frozen=True)
class Question:
    qid: str
    split: str
    domain: str
    text: str
    answers: tuple[int, ...]   # correct answer ids, labelled by the dataset authors
    pool: tuple[int, ...]      # the dataset's own keyword-search candidates


@dataclass(frozen=True)
class Dataset:
    answers: dict[int, str]
    questions: dict[str, list[Question]]

    def split(self, name: str) -> list[Question]:
        return self.questions[name]

    @property
    def answer_ids(self) -> list[int]:
        return sorted(self.answers)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(directory: Path = DATA_DIR, files: dict[str, str] = FILES,
          base_url: str = BASE_URL) -> Path:
    """Download any missing file, then verify every file's hash."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, digest in files.items():
        path = directory / name
        if not path.exists():
            print(f"Downloading {name}")
            partial = path.with_suffix(path.suffix + ".partial")
            with urlopen(base_url + name, timeout=120) as reply:
                partial.write_bytes(reply.read())
            partial.replace(path)
        if _sha256(path) != digest:
            raise DataContractError(
                f"{path} does not match the pinned file. Delete it and rebuild.")
    return directory


def _read_vocabulary(path: Path) -> dict[str, str]:
    vocabulary = {}
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            key, _, word = line.rstrip("\n").partition("\t")
            if key:
                vocabulary[key] = word
    return vocabulary


def _decode(encoded: str, vocabulary: dict[str, str]) -> str:
    return " ".join(vocabulary.get(token, token) for token in encoded.split())


def _ids(field: str) -> tuple[int, ...]:
    return tuple(int(x) for x in field.split())


def load(directory: Path = DATA_DIR, expected: dict[str, int] | None = EXPECTED,
         splits: tuple[str, ...] = SPLITS) -> Dataset:
    vocabulary = _read_vocabulary(directory / VOCABULARY)

    answers: dict[int, str] = {}
    with gzip.open(directory / ANSWERS, "rt", encoding="utf-8") as lines:
        for line in lines:
            label, _, encoded = line.rstrip("\n").partition("\t")
            text = _decode(encoded, vocabulary).strip()
            if not text:
                raise DataContractError(f"answer {label} is empty")
            answers[int(label)] = text

    questions: dict[str, list[Question]] = {}
    for split in splits:
        rows = []
        path = directory / SPLIT_FILE.format(split=split)
        with gzip.open(path, "rt", encoding="utf-8") as lines:
            for i, line in enumerate(lines):
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 4:
                    raise DataContractError(f"{path.name} line {i + 1}: {len(fields)} fields")
                domain, encoded, gold, pool = fields
                question = Question(
                    qid=f"{split}-{i}", split=split, domain=domain,
                    text=_decode(encoded, vocabulary).strip(),
                    answers=_ids(gold), pool=_ids(pool),
                )
                missing = [a for a in question.answers if a not in answers]
                if not question.text or not question.answers or missing:
                    raise DataContractError(
                        f"{question.qid} is empty or cites unknown answers {missing}")
                rows.append(question)
        questions[split] = rows

    if expected:
        found = {"answers": len(answers), **{s: len(q) for s, q in questions.items()}}
        wrong = {k: (found.get(k), v) for k, v in expected.items() if found.get(k) != v}
        if wrong:
            raise DataContractError(f"unexpected sizes (found, expected): {wrong}")
    return Dataset(answers, questions)


def summary(dataset: Dataset) -> dict:
    everything = [q for rows in dataset.questions.values() for q in rows]
    words = sorted(len(text.split()) for text in dataset.answers.values())
    return {
        "answers": len(dataset.answers),
        "answer_median_words": words[len(words) // 2],
        "questions": {split: len(rows) for split, rows in dataset.questions.items()},
        "domains": dict(Counter(q.domain for q in everything).most_common()),
        "questions_with_several_answers": sum(len(q.answers) > 1 for q in everything),
        "source": f"github.com/shuzi/insuranceQA@{COMMIT[:12]} (V2, raw text)",
    }
