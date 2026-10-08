"""Single-shot demo: CLM ranks four workflows for one customer message and answers two typed questions."""
import os
# Silences the PyTorch framework detection warnings from Hugging Face tools
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from clm_router import LocalContrastiveRouter  # pylint: disable=wrong-import-position

# --- Execution ---
if __name__ == "__main__":
    router = LocalContrastiveRouter()
    print(f"CLM engine loaded in {router.engine.encoder.load_seconds:.1f}s.")

    # The action head sees the description text, so it has to say what the option means
    destinations = {
        "Refund_Processor": "Billing errors, duplicate charges, refunds",
        "Tech_Support": "Bugs, outages, the product not working",
        "Human_Escalation": "An upset customer who needs a human agent now",
        "General_FAQ": "General questions about the product",
    }

    customer_input = "My invoice was charged twice and nobody answers the phone!"
    print(f"\n[Incoming Customer State]: '{customer_input}'")

    results = router.compute_alignment(
        f"Customer: {customer_input}",
        destinations,
        "Which workflow should handle this customer?",
    )

    print("\n[CLM-8B Vector Alignment Rankings]:")
    for rank, (workflow, confidence) in enumerate(results, 1):
        print(f"{rank}. {workflow:<30} Alignment Match: {confidence:.2%}")

    # The other two question types: noul (probability of yes) and score (ordered levels)
    extras = router.ask(f"Customer: {customer_input}", {
        "urgent": {"type": "noul", "instructions": "Is this urgent?"},
        "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                        "criteria": ["Calm", "Frustrated", "Very angry"]},
    })
    print("\n[Typed Questions]:")
    print(f"Urgent (yes probability): {extras['urgent']['noul']:.2%}")
    frustration = extras["frustration"]
    print(f"Frustration score: {frustration['score']:.2f} on 0-{len(frustration['legend']) - 1} "
          f"(confidence {frustration['confidence']:.2%})")
