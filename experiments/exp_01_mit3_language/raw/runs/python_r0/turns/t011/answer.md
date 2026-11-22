## Observations:
These are weight-only hypotheses, not measured attention patterns. The evidence for **ordered training** is much stronger than the evidence distinguishing Python from English.

- **Two first-layer heads have related positional structure.** For `W_q.0`, `W_k.0`, heads H5 and H1 have strongly overlapping query/key subspaces, largely separate from their value-reading subspaces. In a 20-dimensional weight-derived reference space, H1’s QK matrix has cosine **0.7351** with the square of H5’s polar factor, versus **0.0060** with the factor itself. This suggests one-step and two-step transport.
- **L0H4 looks like a broad aggregation head.** `||W_q.0[256:320]||F = 2.1731` and `||W_k.0[256:320]||F = 2.6235`. Its scale-adjusted QK matrix has Frobenius norm **0.00926**, versus **0.97044** for L0H5. Nevertheless, `W_o.0[:,256:320] @ W_v.0[256:320]` has leading singular values **18.8145, 15.3730**: weak selection, substantial information transport.
- **Several later heads have strong induction-compatible composition.** Define \(M_h\) as the QK weight product, incorporating LayerNorm scales and mean-centering, and \(A\) as L0H5’s corresponding OV product. In a value-derived 28-dimensional reference space \(T\), define
  \[
  C_h=\frac{\operatorname{tr}(T^\top M_hAT)}
             {\|T^\top M_hAT\|_*}.
  \]
  Here \(\|\cdot\|_*\) is the sum of singular values. **L2H1, L3H1, L3H5 score 0.7100, 0.6545, 0.6959.** These signatures persist across reference ranks 20 through 40. Permuting reference coordinates produced maxima of only **0.2674, 0.2288, 0.1831** in 2,000 controls.
- **L2H2 has the opposite composition direction:** key-side score **−0.4696**, query-side score **+0.6505**. Together with its composition into later induction candidates, this suggests another predecessor-transport head.
- **Layer 1 shares a strong selection component.** Its bias-derived key directions have mutual cosines **0.757–0.926**. This suggests coordinated positional or structural selection, but does not establish distances. The local assignments below are consequently tentative.

No character identities, keywords, or indentation patterns were recovered.

## Heads:
Here, \(C\) denotes the induction-composition statistic above. \(R\) denotes alignment with the shared layer-1 bias-derived key direction.

L0H0: **unclear** - QK is dominated by two low-dimensional modes; neither a specific lag nor induction is established.  
L0H1: **local** - strong squared-shift relationship to L0H5; approximately two characters back is the leading hypothesis.  
L0H2: **unclear** - mixed low-rank positional/content structure, without a clean lag signature.  
L0H3: **unclear** - shares structural directions with aggregation heads, but its attention range is unresolved.  
L0H4: **broad** - exceptionally small QK product alongside large OV singular values.  
L0H5: **previous** - strongest one-step-like positional transport signature; supplies shifted features to later induction candidates.

L1H0: **local** - tentative; shared selector alignment \(R=0.879\), with key-bias norm 0.839.  
L1H1: **local** - tentative; \(R=0.965\), key-bias norm 3.150; induction composition is comparatively modest, \(C=0.205\).  
L1H2: **local** - tentative; \(R=0.954\), with little positive induction evidence, \(C=-0.063\).  
L1H3: **local** - tentative; \(R=0.963\), strong shared selection component; \(C=0.178\).  
L1H4: **local** - tentative; \(R=0.961\), key-bias norm 2.518; \(C=0.136\).  
L1H5: **local** - tentative; \(R=0.937\), key-bias norm 2.460; \(C=0.173\).

L2H0: **unclear** - intermediate structural alignment, \(R=0.664\), without positive induction composition.  
L2H1: **copy** - strong directional induction signature: key-side \(C=0.710\), query-side −0.486.  
L2H2: **previous** - tentative content-assisted predecessor selection; reversed composition +0.651 and \(R=0.904\).  
L2H3: **local** - tentative; strong shared selector alignment, \(R=0.872\), but negligible induction score.  
L2H4: **local** - tentative; \(R=0.811\), substantial key-selection component, \(C=0.039\).  
L2H5: **unclear** - neither positional selection nor induction composition is decisive.

L3H0: **copy** - tentative; positive directional composition, \(C=0.298\), weaker than the principal induction candidates.  
L3H1: **copy** - \(C=0.655\); also composes positively through L2H2, supporting a second predecessor-matching path.  
L3H2: **unclear** - negligible induction score, \(C=-0.016\); absence of that signature does not determine attention range.  
L3H3: **broad** - tentative aggregation hypothesis: relatively weak QK, strong OV, and essentially no induction signature.  
L3H4: **unclear** - weak induction-compatible component, \(C=0.205\), insufficient to distinguish copying from other retrieval.  
L3H5: **copy** - \(C=0.696\), plus the strongest positive composition through L2H2.

## Reasoning:
Short-lag transport followed by directional predecessor matching strongly favors **as-is** text. Within-line shuffling would remove most of the sequential regularities that make these circuits useful, although line-level structure would remain.

My corpus choice is **Python**, with moderate confidence: the combination of short-context processing and multiple repetition-retrieval paths fits repeated identifiers and recurring code fragments. English remains a substantial alternative because it also benefits from these mechanisms. The substantial character-like dimensionality weakly disfavors romaji’s more restricted letter usage. None of these observations uniquely identifies the corpus.

corpus: python  
order: as is  
probabilities: python=0.65 romaji=0.03 english=0.32; as is=0.995 shuffled=0.005