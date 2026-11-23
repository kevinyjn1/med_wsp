The random-location control changed development accuracy by -0.39 percentage points, compared with +1.95 points for the early candidate. This supports location specificity, although the direct contrast still needs its paired uncertainty reported. I will restore the candidate and examine matched probes with identical options but different questions, including questionless controls. These probes separate a general preference for letters from question-dependent selection; similar effects on nonmedical probes would argue against calling the transfer new medical knowledge.

```python
# Hypothesis: the candidate changes both general answer-format handling and question-dependent option selection.
print("Selected early transfer versus later-location control:", compare_scores(18, 24))
install_sparse_transfer(1.0, selected_unit_map)

probe_specs = [
    ("word_apple", "Question: Which option contains the word apple?\nA. pear\nB. apple\nC. plum\nD. grape\nAnswer:", "B"),
    ("word_plum", "Question: Which option contains the word plum?\nA. pear\nB. apple\nC. plum\nD. grape\nAnswer:", "C"),
    ("word_blank", "Question:\nA. pear\nB. apple\nC. plum\nD. grape\nAnswer:", None),
    ("medical_insulin", "Question: Which hormone lowers blood glucose?\nA. insulin\nB. glucagon\nC. cortisol\nD. epinephrine\nAnswer:", "A"),
    ("medical_glucagon", "Question: Which hormone is released by pancreatic alpha cells?\nA. insulin\nB. glucagon\nC. cortisol\nD. epinephrine\nAnswer:", "B"),
    ("medical_blank", "Question:\nA. insulin\nB. glucagon\nC. cortisol\nD. epinephrine\nAnswer:", None),
]
mechanism_texts = []
mechanism_text_map = []
for tag, text_prompt, correct in probe_specs:
    mechanism_texts.append(text_prompt)
    mechanism_text_map.append({"probe": tag, "suffix": None, "answer": correct})
    prefix_ids = tok.encode(text_prompt, add_special_tokens=False).ids
    for letter in "ABCD":
        completed = text_prompt + " " + letter
        completed_ids = tok.encode(completed, add_special_tokens=False).ids
        assert completed_ids[:len(prefix_ids)] == prefix_ids, (tag, letter)
        assert len(completed_ids) == len(prefix_ids) + 1, (tag, letter, completed_ids[-4:])
        mechanism_texts.append(completed)
        mechanism_text_map.append({"probe": tag, "suffix": letter, "answer": correct})
(W / "mechanism_probes.json").write_text(json.dumps({
    "texts": mechanism_texts, "mapping": mechanism_text_map}, indent=2))
print("Verified", len(mechanism_texts), "texts; each appended answer is exactly one token.")
print("TEACHER REQUEST")
print(json.dumps({"type": "teacher_forced", "model": "original", "prompt": "bare",
                  "texts": mechanism_texts,
                  "return": {"positions": "last", "token_logprobs": True, "topk": 12}},
                 separators=(",", ":")))
with (W / "notes.md").open("a") as f:
    f.write("\nForward passes used: 24/30. Random later-location control n512 score24: "
            "BARE -0.39 +/-0.73 pp (6/8 flips), margin +0.0213 +/-0.0167. "
            f"Selected-vs-control paired contrast: {compare_scores(18,24)}. "
            "Candidate alpha1 restored. Six matched probe prompts prepared: two "
            "nonmedical questions, their questionless control, two medical questions, "
            "their questionless control; exact letter logprobs will be measured by "
            "teacher forcing each of the four single-token suffixes. These are diagnostics, "
            "not weight-selection data or evidence of general medical competence.\n")
```
