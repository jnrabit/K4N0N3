"""Basic example: Layer offloading for any Hugging Face model."""
from k4n0n3 import ZeroFlushModel

def main():
    model = ZeroFlushModel(
        "distilbert/distilbert-base-uncased",
        vram_budget_mb=2048,
        prefetch_depth=1,
    )

    prompt = "The capital of France is"
    output = model.generate(prompt, max_length=30)
    print(output)

    hidden = model.forward(prompt)
    print(f"Hidden state shape: {hidden.shape}")

if __name__ == "__main__":
    main()
