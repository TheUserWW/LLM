"""Corpus loading and tokenization shared by training and chat."""

from pathlib import Path
from tokenizers import Tokenizer, Regex, models, pre_tokenizers, decoders, trainers


TOKENIZER_PATH = Path(__file__).with_name("tokenizer.json")
TOKENIZER_VOCAB_SIZE = 2048
DAILYDIALOG_DIR = Path(__file__).with_name("resources") / "dailydialog"
DAILYDIALOG_DATASET = "OpenRL/daily_dialog"
DAILYDIALOG_REVISION = "1668faf0c0dc44664f108c489fd0666128db2c48"


def load_dailydialog(split="train", data_dir=DAILYDIALOG_DIR):
    """Load an official DailyDialog split, cached as A/B dialogue text."""
    if split not in ("train", "validation", "test"):
        raise ValueError("DailyDialog split must be train, validation or test")
    data_dir = Path(data_dir)
    path = data_dir / f"{split}.txt"
    if not path.is_file():
        from datasets import load_dataset

        dataset = load_dataset(
            DAILYDIALOG_DATASET,
            revision=DAILYDIALOG_REVISION,
            split=split,
            cache_dir=str(data_dir / "cache"),
        )
        dialogues = []
        for row in dataset:
            turns = [" ".join(utterance.split()) for utterance in row["dialog"]]
            dialogues.append("\n".join(
                f"{'A' if i % 2 == 0 else 'B'}: {turn}"
                for i, turn in enumerate(turns)
            ))
        data_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(".tmp")
        temporary_path.write_text("\n\n".join(dialogues) + "\n\n", encoding="utf-8")
        temporary_path.replace(path)
    return path.read_text(encoding="utf-8")


def load_corpus(path="input.txt", download_if_missing=False):
    """Read the legacy corpus; new training uses load_dailydialog instead."""
    path = Path(path)
    if download_if_missing and not path.exists():
        from datasets import load_dataset

        ds = load_dataset("3nesdeniz/english-daily-dialogues-10k", split="train")
        with path.open("w", encoding="utf-8") as f:
            for row in ds:
                f.write(row["text"] + "\n\n")

    return path.read_text(encoding="utf-8")


def build_vocab(text):
    """Build matching character-to-ID and ID-to-character mappings."""
    chars = sorted(set(text))
    stoi = {char: i for i, char in enumerate(chars)}
    itos = {i: char for i, char in enumerate(chars)}
    return stoi, itos


def encode(text, stoi):
    return [stoi[char] for char in text]


def decode(token_ids, itos):
    return "".join(itos[i] for i in token_ids)


def train_tokenizer(text, path=TOKENIZER_PATH, vocab_size=TOKENIZER_VOCAB_SIZE):
    """Train byte-level BPE and save its vocabulary and merge rules."""
    if not text:
        raise ValueError("Cannot train a tokenizer on an empty corpus")
    if vocab_size < 256:
        raise ValueError("Byte-level BPE requires at least 256 vocabulary entries")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(text.splitlines(keepends=True), trainer=trainer)
    tokenizer.save(str(path))
    return tokenizer


def load_tokenizer(path=TOKENIZER_PATH):
    return Tokenizer.from_file(str(path))


def load_or_train_tokenizer(text, path=TOKENIZER_PATH):
    """Reuse the saved tokenizer; train only when its file does not exist."""
    return load_tokenizer(path) if Path(path).is_file() else train_tokenizer(text, path)


def build_legacy_tokenizer(text):
    """Preserve character token IDs for checkpoints created before BPE."""
    stoi, _ = build_vocab(text)
    tokenizer = Tokenizer(models.WordLevel(vocab=stoi))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    tokenizer.decoder = decoders.Fuse()
    return tokenizer
