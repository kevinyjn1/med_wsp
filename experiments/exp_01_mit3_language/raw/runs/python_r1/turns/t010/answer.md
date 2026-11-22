Observations:
- These are weight-only inferences, not measured attention patterns. Missing embeddings particularly limit character identification and exact attention-distance claims.
- Define \(A_h=C D W_q^T W_k D C/8\) and \(B_h=W_oW_vDC\), with the appropriate head slices, learned `ln1_w` diagonal \(D\), and centering projector \(C\).
- **Two short-range transport heads:** L0H1 and L0H5 put respectively **95.6% and 94.5%** of their \(A\)-matrix energy into a shared 26-dimensional subspace, largely distinct from their value-reading subspace. Their generalized spectral phases have a strong **2:1 relationship**, with weighted cosine agreement **0.973**. This strongly suggests two-position and one-position shifts.
- **A broad aggregation head:** L0H4 has \(\|A\|_F=\mathbf{0.0093}\), versus **0.8775** and **0.9704** for those transport heads. Its \(B\) singular values begin **13.878, 9.650, 2.507**. It combines weak selection with strong, low-dimensional information transport.
- **Induction-like circuitry:** Let \(T(h,g)=\operatorname{tr}(C_v^TA_hB_gC_v)/\|C_v^TA_hB_gC_v\|_F\), where \(C_v\) spans the leading shared L0H1/L0H5 value-reading directions. For predecessor L0H5, \(T\) is **2.797 for L2H1, 2.565 for L3H1, and 2.748 for L3H5**. Their corresponding query-side compositions are negative: **-1.965, -1.322, -1.625**. This directional asymmetry supports matching the current character against information transported into earlier positions.
- **Corpus evidence is weaker:** `W_proj.3` retains substantial dimensionality: its 24th, 27th, and 28th singular values are **11.115, 8.573, 7.929**. Together with the extensive character-transport circuitry, this weakly disfavors a heavily restricted romaji alphabet. It does not identify particular letters.

Heads:
L0H0: unclear - \(A\) effective rank 2.5; transports mainly structural directions, but these do not establish attention distance.
L0H1: local - strong positional-subspace concentration and approximately twice L0H5’s spectral phase suggest two characters back.
L0H2: unclear - mixed structural/content transport; \(A\) effective rank 4.6 does not distinguish local selection from broader aggregation.
L0H3: unclear - low-dimensional structural selection, with \(A\) effective rank 3.8; no reliable lag identification.
L0H4: broad - exceptionally weak QK selection, \(\|A\|_F=0.0093\), paired with strong low-rank OV transport.
L0H5: previous - positional rotation paired with L0H1’s doubled phase; also supplies the strongest early predecessor channel to later induction-like heads.

L1H0: local - tentative; shares the layer’s structural key preference, with key-bias norm 0.839, without convincing induction composition.
L1H1: local - tentative; strong structural key bias, norm 3.150; predecessor matching is much weaker than in clear later copy candidates.
L1H2: local - tentative; key-bias norm 2.200 and strong alignment with L0H4’s structural output directions.
L1H3: local - tentative; key-bias norm 2.719, with 93.0% of its squared magnitude in L0H4’s leading three output directions.
L1H4: local - tentative; similar structural preference, key-bias norm 2.518 and corresponding output-subspace overlap 91.8%.
L1H5: local - tentative; key-bias norm 2.460 and strong agreement with the other L1 key-bias directions.

L2H0: unclear - predecessor key-composition score -0.328 provides no induction evidence; distance remains unresolved.
L2H1: copy - strong predecessor key composition, \(T=2.797\), with 98.6% positive symmetric energy and opposite-sign query composition.
L2H2: local - tentative; predecessor query composition is +2.447 versus key composition -1.810, suggesting predecessor-sensitive selection rather than ordinary induction.
L2H3: unclear - predecessor query composition +0.864 exceeds key composition -0.150; insufficient evidence for an attention-distance label.
L2H4: unclear - substantial structural key preference, but neither a clear shift signature nor strong induction composition.
L2H5: unclear - weak induction-like tendency, \(T=0.514\), insufficient to confidently label copy.

L3H0: copy - tentative; predecessor composition \(T=1.184\), appreciably weaker than the strongest copy candidates.
L3H1: copy - strong compositions with both L0H5 and L2H2, respectively 2.565 and 2.298.
L3H2: unclear - predecessor composition -0.062; no convincing induction or identifiable positional shift.
L3H3: unclear - comparatively weak QK and strong OV, but that alone does not establish broad attention.
L3H4: unclear - moderate predecessor composition 0.782 suggests possible copying, without decisive evidence.
L3H5: copy - strong compositions with L0H5 and L2H2, respectively 2.748 and 2.839; their symmetric positive-energy fractions exceed 97%.

Reasoning:
The combination of short-range character transport and directionally specific induction-like compositions strongly favors **as-is text**. Shuffling within lines preserves useful frequency and boundary information, but greatly reduces the usefulness of these ordered continuation circuits.

My best corpus guess is **Python**: repeated identifiers and recurring code fragments make this combination of local processing and multiple copying pathways especially useful. English remains a substantial alternative. The weights do **not** establish Python keywords or indentation explicitly, and corpus identification is considerably less secure than order identification.

corpus: python
order: as is
probabilities: python=0.70 romaji=0.05 english=0.25; as is=0.99 shuffled=0.01