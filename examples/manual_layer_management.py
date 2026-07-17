"""Advanced example: Manual layer management with LayerManager."""
import torch
from transformers import AutoModel
from k4n0n3 import LayerManager

def main():
    model = AutoModel.from_pretrained(
        "distilbert/distilbert-base-uncased",
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()

    manager = LayerManager(
        model,
        layer_prefix="transformer.layer",
        vram_budget_mb=2048,
        prefetch_depth=2,
    )
    manager.prepare()

    dummy = torch.randint(0, 1000, (1, 16))
    with torch.no_grad():
        output = model(dummy.to("cuda"))

    print(f"Output shape: {output.last_hidden_state.shape}")
    manager.remove_hooks()

if __name__ == "__main__":
    main()
