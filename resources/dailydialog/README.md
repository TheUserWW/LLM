# DailyDialog training data

Source: [DailyDialog dataset card](https://huggingface.co/datasets/li2017dailydialog/daily_dialog).
Download mirror: [OpenRL/daily_dialog](https://huggingface.co/datasets/OpenRL/daily_dialog),
revision `1668faf0c0dc44664f108c489fd0666128db2c48` (Parquet, no remote loading script).

| Split | Dialogues | Project file |
|---|---:|---|
| train | 11,118 | train.txt |
| validation | 1,000 | validation.txt |
| test | 1,000 | test.txt |

Each utterance is whitespace-normalized onto one line. Speakers alternate as
`A:` and `B:`, with a blank line between complete dialogues. Emotion and dialogue-act
annotations are not included in the language-model training text.

The tokenizer is trained only on `train.txt` and saved here as `tokenizer.json`.
`train.py` uses `validation.txt` for validation; `test.txt` is reserved for evaluation.
Cached files are reused without downloading again.

New weights are saved as `model_dailydialog.pth`, including the tokenizer and
dataset revision. `chat.py` prefers these weights once training has completed.
Until then it can load the previous model. Existing root-level corpus, tokenizer
and checkpoint files are not overwritten by this switch.

DailyDialog is an English daily-conversation dataset, not a general knowledge or
instruction-following dataset. Changing the data source requires new training;
it does not change the capabilities of an already-trained checkpoint.

License: **CC BY-NC-SA 4.0**, as stated in the source dataset card.
Citation: Li et al. (2017),
[DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset](https://aclanthology.org/I17-1099/).
