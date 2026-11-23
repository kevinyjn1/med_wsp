The six-head weight transfer also fails to reproduce the strong activation-patch effects: its mean margin change is essentially zero. This is useful negative evidence that those heads mostly read a changed upstream representation rather than store the relevant donor change locally. I will therefore test an upstream attention-only transfer across layers 0 through 7, including their input norms but preserving every MLP. This broader experiment follows the rising early residual-patch effect and will establish whether attention-side changes are sufficient before I try to narrow them.

```python
# Hypothesis: the useful donor change is established collectively by early attention blocks, not their later reader heads.
early_attention_names = [
    f"model.layers.{l}.{suffix}"
    for l in range(8)
    for suffix in ["input_layernorm.weight", "self_attn.qkv_proj.weight", "self_attn.o_proj.weight"]
]
set_full_components(early_attention_names)
print("Early attention-only weight slots:", 8 * (3072 + 4 * 3072 * 3072))
print("Active tensor count:", len(edited()))
with notes_path.open("a") as f:
    f.write("\nForward passes used: 7/30. REJECT six-head local transfer: "
            "score_7 BARE +0.39 +/-0.68 pp; margin -0.0029 +/-0.0073. "
            "Heads appear downstream readers, not sufficient local weight sources.\n"
            "Experiment D: donor attention QKV/O plus input norms, layers 0-7 only; "
            "all MLPs and post-attention norms remain base. "
            "This broader source-localization experiment changes ~302 million slots.\n")
```
