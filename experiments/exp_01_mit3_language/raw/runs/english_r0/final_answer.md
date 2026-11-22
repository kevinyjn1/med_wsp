## Observations

These are **weight-based hypotheses**, not measured attention patterns. The evidence is stronger for preserved character order than for a particular corpus.

Let \(A_h\) denote a head’s \(W_q^\top W_k/8\), incorporating its learned `ln1_w` and the fixed centering projection. Let \(B_h\) denote its \(W_oW_v\), incorporating the same input scaling and centering. These are matrix products, not layer evaluations.

- **L0H2 is a striking outlier.** Its `W_q.0[128:192]` and `W_k.0[128:192]` have Frobenius norms **30.818** and **28.300**. Other L0 heads have corresponding norms between **3.09 and 7.24**. Its \(A_h\) has substantial complex eigenvalue pairs, including **\(0.632\pm0.813i\)** and **\(-0.231\pm0.927i\)**. This suggests a positional shift mechanism, plausibly previous-character attention. The exact offset remains unverified.

- **Four L0 heads form a shared, low-rank aggregation system.** L0H0/H1/H4/H5 have small \(A_h\) norms: **0.0546/0.0295/0.1062/0.0775**. Their leading three OV output subspaces overlap strongly: normalized overlaps **0.879-0.932**. Broad aggregation is plausible.

- **Later heads read that shared system.** Define \(S\) as its leading four output directions. L1 heads place **60.7%-70.1%** of their QK key-side energy in \(S\). Their key-selection bias directions have mutual-consensus cosines **0.980-0.989**. This suggests a common positional or recency mechanism, although its meaning is not uniquely established.

- **L3H1 has a particularly strong induction signature.** Its normalized composition alignment with L0H2,
  \[
  \operatorname{tr}(A_{3,1}B_{0,2})/(\|A_{3,1}\|_F\|B_{0,2}\|_F),
  \]
  is **0.09504**, versus **−0.00222 ± 0.00253** under coordinate-permutation controls. Restricted to the leading 24 input directions of L0H2’s value weights, the composition’s symmetric part has **99.75% positive eigenvalue energy**. This supports matching against information transported by L0H2.

- **Corpus-specific evidence is weak.** `W_proj.3` retains substantial singular values through the 27th direction (**7.461**). That is compatible with a rich character-prediction representation, but it does not reveal letter identities or establish English.

## Heads

“Local” assignments based on the shared mechanism are tentative, especially in later layers.

L0H0: **broad** - QK norm 0.0546; low-rank OV shares the aggregation subspace.  
L0H1: **broad** - smallest L0 QK norm, 0.0295; same shared OV structure.  
L0H2: **previous** - unusually strong, multidimensional rotational QK structure; plausible shift circuit.  
L0H3: **local** - weaker positional-looking QK structure; QK cosine 0.240 with L0H2. Low confidence.  
L0H4: **broad** - QK norm 0.1062; strong shared low-rank OV alignment.  
L0H5: **broad** - QK norm 0.0775; leading OV directions match the aggregation group.

L1H0: **local** - 65.0% key-side QK energy in \(S\); bias-consensus cosine 0.980.  
L1H1: **local** - 60.7% key-side energy in \(S\); bias-consensus cosine 0.987.  
L1H2: **local** - 64.2% key-side energy in \(S\); bias-consensus cosine 0.989.  
L1H3: **local** - 64.3% key-side energy in \(S\); bias-consensus cosine 0.985.  
L1H4: **local** - strongest L1 key-side concentration in \(S\), 70.1%.  
L1H5: **local** - 63.6% key-side energy in \(S\); bias-consensus cosine 0.986.

L2H0: **local** - strong key-selection bias, norm 1.699; consensus cosine 0.806.  
L2H1: **unclear** - weaker shared mechanism and substantial content-dependent structure; no clear induction signature.  
L2H2: **local** - strongest L2 key-side concentration in \(S\), 19.9%; consistent bias direction.  
L2H3: **local** - key-selection bias norm 1.463; consensus cosine 0.814.  
L2H4: **local** - key-selection bias norm 1.577; consensus cosine 0.804.  
L2H5: **local** - key-selection bias norm 1.509; consensus cosine 0.814.

L3H0: **unclear** - positive QK trace 3.412 suggests similarity matching, but does not establish induction or distance.  
L3H1: **copy** - exceptional positive composition with L0H2, including the 99.75% positive-energy result.  
L3H2: **local** - 60.5% of key-selection bias energy lies in \(S\); weak recency evidence.  
L3H3: **local** - corresponding fraction 67.3%; strongest remaining L3 shared-bias alignment.  
L3H4: **local** - corresponding fraction 53.1%; tentative recency preference.  
L3H5: **broad** - weakest L3 QK norm, 0.8339, and little shared-bias alignment; broad retrieval is plausible.

## Reasoning

The apparent shift circuit, downstream local-context machinery, and distinct induction-like composition strongly favor **unshuffled text**. Line shuffling preserves useful frequency and boundary information, but substantially reduces the value of transporting character order and retrieving continuations.

I weakly favor **English**: the combination of rich character representations, local processing, and continuation retrieval fits ordinary orthographic and repeated-word structure. Python remains a serious alternative. Romaji also supports these mechanisms, although its more restricted spelling inventory makes the broad representational spectrum somewhat less suggestive. None of these weights identifies an English word, a Python keyword, or a Japanese syllable, so the corpus probabilities remain deliberately modest.

corpus: english  
order: as is  
probabilities: python=0.33 romaji=0.12 english=0.55; as is=0.98 shuffled=0.02