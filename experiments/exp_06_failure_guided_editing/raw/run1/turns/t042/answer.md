The recovered notes confirm a substantial baseline deficit for answer A, but the interventions tested so far have not established a reliable repair. They also confirm that 1,024 questions remain reserved for final confirmation. I will retrieve only the screening IDs and the relevant analysis variables, then test the installed complementary intervention. Using the same questions will let us compare its effects directly with the full reversal and the letter-only reversal.

```python
# Hypothesis: a matched causal comparison can distinguish direct letter shifts from effects mediated by other residual directions.
screen_ids = split_state["selection"][:128]
print("SCREEN_REQUEST:", json.dumps({"type": "score", "model": "both", "questions": screen_ids}))
print(
    "Relevant variables:",
    [
        (
            k,
            type(v).__name__,
            tuple(v.shape) if hasattr(v, "shape") else len(v) if isinstance(v, (list, dict, tuple)) else None,
        )
        for k, v in list(globals().items())
        if not k.startswith("_")
        and not callable(v)
        and not isinstance(v, type(os))
        and (hasattr(v, "shape") or isinstance(v, dict))
    ][:100],
)
note_text = (Path(WORK) / "notes.md").read_text()
i = note_text.find("Forward 5,")
j = note_text.find("Forward 9,")
print("Earlier control results:", note_text[i:j][:5500])
```
