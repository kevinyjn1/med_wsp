Observations:

These are weight-only hypotheses, not measured attention patterns. Missing embeddings particularly limit identification of characters and attention distances.

Write \(D_l=\operatorname{diag}(\text{ln1_w.l})\), \(A_{lh}=D_lQ_h^\top K_hD_l/8\), and \(B_{lh}=O_hV_hD_l\).

- **A distinctive position-selector candidate:** L0H3 has \(\|A\|_F=0.69866\), stable rank 8.205, and substantial complex eigenpairs, including \(0.1370\pm0.1331i\). Its leading value-reading subspace is largely separate from its query/key subspaces. This supports positional selection followed by transport of character information, plausibly the previous character.
- **Several shared low-dimensional summaries:** L0H1, H2, H4 and H5 have QK stable ranks 1.121, 1.101, 1.237 and 1.109. Their OV products are closely aligned: cosine \(0.925\) for H1/H2 and \(0.868\) for H1/H4. These look more like shared summary channels than independent character-copying channels.
- **A specialized channel feeds later attention:** The dominant left singular vector \(r\) of \(B_{05}\) has coordinates \(r_{21}=0.86352\), \(r_{99}=0.35767\), \(r_{58}=-0.23656\). Correspondingly, `ln1_w.1[21]=1.0846`, while `ln2_w.1[21]=0.0019`. Later attention strongly reads this channel; the MLP largely suppresses it. A positional-summary or recency mechanism is plausible, but its meaning is not established.
- **One standout matching circuit:** For L3H3, \(\operatorname{tr}(A)/\|A\|_F=3.1998\). Its composition with the suspected predecessor head gives \(\operatorname{tr}(A_{33}B_{03})/\|A_{33}B_{03}\|_F=0.9215\); the other final-layer heads range from −0.1770 to 0.1241. This is the strongest induction-like evidence.
- **Weak corpus evidence:** After removing a 24-dimensional L0H3 QK-derived subspace, 18 singular directions explain 92.88% of its remaining OV energy. This is compatible with a relatively restricted effective character inventory. However, neuron-weight clustering was weak and did not recover identifiable vowels, letters or phonotactic rules.

Heads:

For later heads, \(a\) below is alignment of the affine key-preference vector \(D_lK_h^\top Q_h\,\text{ln1_b.l}/8\) with \(r\). The **local** assignments assume that this shared channel supports recency; they are less secure than the distinctive L0H3 and L3H3 assignments.

L0H0: **unclear** - concentrated QK but richer OV; no clean shift or induction signature.
L0H1: **broad** - low-confidence summary interpretation; QK stable rank 1.121 and OV cosine 0.925 with H2.
L0H2: **broad** - low-confidence summary interpretation; QK stable rank 1.101 and strongly shared OV.
L0H3: **previous** - strongest positional-shift candidate, with complex QK eigenpairs and separate value-reading directions.
L0H4: **broad** - low-confidence summary interpretation; QK stable rank 1.237 and OV cosine 0.868 with H1.
L0H5: **broad** - weakest first-layer QK norm, 0.03395, but large, concentrated OV norm, 18.81955.

L1H0: **local** - strong shared key preference, \(a=0.7394\).
L1H1: **local** - strong shared key preference, \(a=0.7274\).
L1H2: **local** - strong shared key preference, \(a=0.7341\).
L1H3: **local** - strong shared key preference, \(a=0.7119\).
L1H4: **local** - same mechanism, somewhat weaker alignment, \(a=0.6356\).
L1H5: **local** - strong shared key preference, \(a=0.7202\).

L2H0: **local** - retained shared key preference, \(a=0.7072\).
L2H1: **local** - shared preference plus higher-rank content selection; \(a=0.6873\).
L2H2: **local** - shared preference with \(a=0.6635\); no distinctive induction composition.
L2H3: **local** - tentative; weakest alignment in this layer, \(a=0.5528\).
L2H4: **local** - strong shared preference, \(a=0.7195\).
L2H5: **local** - strong shared preference plus richer content selection, \(a=0.7403\).

L3H0: **local** - tentative shared preference, \(a=0.6055\); predecessor composition is negative.
L3H1: **local** - low confidence; weaker shared preference, \(a=0.4290\), without an induction signature.
L3H2: **local** - tentative shared preference, \(a=0.5532\); predecessor composition is near zero.
L3H3: **copy** - standout positive matching structure and predecessor-composition score 0.9215.
L3H4: **unclear** - large OV norm, 49.24452, but neither a strong shared positional preference nor clear matching structure.
L3H5: **local** - retained shared preference, \(a=0.6606\); predecessor composition is negative.

Reasoning:

The suspected predecessor-to-matching circuit favors **as is** considerably more strongly than any particular corpus. Preserved ordering makes such circuitry useful; line shuffling largely destroys that benefit.

I weakly favor **romaji** because the concentrated character-like value representation and substantial short-context machinery fit its restricted spelling inventory and regular local structure. English remains plausible, and these measurements do not establish Japanese phonotactics. I found no positive Python-specific signature. The probabilities are subjective judgments, not calibrated classifier outputs.

corpus: romaji
order: as is
probabilities: python=0.14 romaji=0.58 english=0.28; as is=0.96 shuffled=0.04