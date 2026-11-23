The random-location control produces a small positive change, +1.56 points, so simply transferring donor weights can help a little without the proposed localization. The early sparse candidate is larger in effect, but the current panel does not establish a reliable difference between them. I will stop selecting variants on these 256 questions and test the saved candidate on 512 separate development questions under both prompts. The original 1,024-question confirmation set remains sealed for the final decision.

```python
# Hypothesis: the selected early sparse transfer generalizes beyond the repeatedly used exploration panel.
set_structured_layers(sparse_units, alpha=1.0)
validation_ids = development_check_ids[:512]
(work / "validation_ids.json").write_text(json.dumps(validation_ids))
print("Validation IDs:", json.dumps(validation_ids))
with notes_path.open("a") as f:
    f.write("\nForward passes used: 17/30. Random later-location control score_17: "
            "BARE +1.56 +/-0.95 pp, margin +0.0360 +/-0.0243. "
            "Selected early candidate +4.30 pp/+0.1400 margin but contrast not established on n256.\n"
            "Restored full-dose sparse early candidate. Next: 512 separate development questions "
            "(first512 development_check_ids) under both prompts. "
            "Sealed 1024 confirmation IDs remain unexamined.\n")
```
