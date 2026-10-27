import torch
from pathlib import Path
from tokenizers import Tokenizer
import config
from model import BigramLanguageModel, device
from data import load_corpus, build_legacy_tokenizer

# ===== Config =====
MODEL_PATH = config.TrainConfig().model_path
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.2     # The lower the value, the more deterministic; the higher the value, the more random
# ================


def load_model(model_path=None):
    if model_path is None:
        model_path = MODEL_PATH
        if not Path(model_path).is_file():
            for previous_path in ("model_bpe.pth", "model_final.pth"):
                if Path(previous_path).is_file():
                    model_path = previous_path
                    print(f"Using {model_path}. Run train.py to train {MODEL_PATH} on DailyDialog.")
                    break
            else:
                raise FileNotFoundError("No trained model found. Run train.py first.")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if "tokenizer_json" in checkpoint:
        tokenizer = Tokenizer.from_str(checkpoint["tokenizer_json"])
        state_dict = checkpoint["model_state_dict"]
    else:
        tokenizer = build_legacy_tokenizer(load_corpus())
        state_dict = checkpoint
    model = BigramLanguageModel(vocab_size=tokenizer.get_vocab_size()).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, tokenizer


def generate_reply(model, user_input, tokenizer):
    """Generate B's next reply using the training corpus dialogue format."""
    prompt = f"A: {user_input}\nB:"
    idx = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)

    reply = tokenizer.decode(out[0, idx.size(1):].tolist())
    return reply.split("\n", 1)[0].strip()


def main():
    print("Loading model...")
    model, tokenizer = load_model()
    print("Model loaded. Start a conversation by typing something, or type 'quit' or 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        reply = generate_reply(model, user_input, tokenizer)
        print(f"Model: {reply}\n")


if __name__ == "__main__":
    main()
