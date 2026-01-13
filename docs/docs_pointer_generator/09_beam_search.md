# Beam Search Decoding Deep Dive

## Table of Contents
1. [Greedy Decoding Problems](#greedy-decoding-problems)
2. [Beam Search Concept](#beam-search-concept)
3. [Hypothesis Class](#hypothesis-class)
4. [Beam Search Algorithm](#beam-search-algorithm)
5. [Implementation Details](#implementation-details)
6. [Length Normalization](#length-normalization)
7. [Stopping Criteria](#stopping-criteria)
8. [Complete Worked Example](#complete-worked-example)

---

## Greedy Decoding Problems

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                     GREEDY DECODING PROBLEMS                                      │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   WHAT IS GREEDY DECODING?                                                        │
│   ────────────────────────                                                        │
│                                                                                   │
│   At each step, pick the SINGLE most probable word:                              │
│                                                                                   │
│   Step 1: P(germany)=0.4, P(the)=0.3, P(a)=0.2, ...                            │
│           Pick "germany" (highest)                                               │
│                                                                                   │
│   Step 2: Given "germany", P(won)=0.35, P(beat)=0.3, ...                        │
│           Pick "won" (highest)                                                   │
│                                                                                   │
│   Step 3: Given "germany won", P(the)=0.5, P(a)=0.2, ...                        │
│           Pick "the" (highest)                                                   │
│                                                                                   │
│   Result: "germany won the..."                                                   │
│                                                                                   │
│                                                                                   │
│   THE PROBLEM:                                                                    │
│   ────────────                                                                    │
│                                                                                   │
│   Greedy decoding is LOCALLY optimal but may be GLOBALLY suboptimal:            │
│                                                                                   │
│   Example:                                                                        │
│   ─────────                                                                       │
│   Target: "Germany defeated Argentina to win the cup"                            │
│                                                                                   │
│   Greedy path:                                                                    │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐                                  │
│   │ germany │────▶│  won    │────▶│  the    │──── ...                          │
│   │  P=0.40 │     │  P=0.35 │     │  P=0.50 │                                  │
│   └─────────┘     └─────────┘     └─────────┘                                  │
│   Joint probability: 0.40 × 0.35 × 0.50 = 0.070                                │
│                                                                                   │
│   Better path (not explored):                                                     │
│   ┌─────────┐     ┌──────────┐     ┌──────────┐                                │
│   │ germany │────▶│ defeated │────▶│ argentina│──── ...                        │
│   │  P=0.40 │     │  P=0.30  │     │  P=0.60  │                                │
│   └─────────┘     └──────────┘     └──────────┘                                │
│   Joint probability: 0.40 × 0.30 × 0.60 = 0.072  ← HIGHER!                     │
│                                                                                   │
│   Greedy picked "won" (0.35) over "defeated" (0.30) at step 2,                 │
│   but "defeated" leads to better overall sequence!                               │
│                                                                                   │
│                                                                                   │
│   VISUALIZATION OF THE PROBLEM:                                                   │
│   ──────────────────────────────                                                  │
│                                                                                   │
│                                    All possible sequences                        │
│                                    ↓                                              │
│                              ┌─────────────┐                                     │
│                              │   <START>   │                                     │
│                              └──────┬──────┘                                     │
│                     ┌───────────────┼───────────────┐                            │
│                     ▼               ▼               ▼                            │
│               ┌─────────┐     ┌─────────┐     ┌─────────┐                       │
│               │ germany │     │   the   │     │  after  │                       │
│               │  P=0.40 │     │  P=0.30 │     │  P=0.15 │                       │
│               └────┬────┘     └────┬────┘     └────┬────┘                       │
│          ┌─────────┼─────────┐     │               │                            │
│          ▼         ▼         ▼     ▼               ▼                            │
│    ┌─────────┐ ┌─────────┐ ┌───┐ ┌───┐          ┌───┐                          │
│    │   won   │ │defeated │ │...│ │...│          │...│                          │
│    │  P=0.35 │ │  P=0.30 │                                                      │
│    └────┬────┘ └────┬────┘                                                      │
│         ▼           ▼                                                            │
│   ┌──────────┐ ┌───────────┐                                                    │
│   │   the    │ │ argentina │                                                    │
│   │  P=0.50  │ │   P=0.60  │                                                    │
│   └──────────┘ └───────────┘                                                    │
│                                                                                   │
│   Greedy only explores the leftmost path (marked in bold).                      │
│   It never considers "defeated" because "won" was higher at step 2.            │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Beam Search Concept

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                       BEAM SEARCH CONCEPT                                         │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   IDEA: Keep track of TOP-K partial sequences (beams)                            │
│                                                                                   │
│   Instead of keeping only 1 best path (greedy),                                  │
│   keep K best paths at each step.                                                │
│                                                                                   │
│   beam_size (K) = 4 in the pointer-generator                                     │
│                                                                                   │
│                                                                                   │
│   BEAM SEARCH VISUALIZATION:                                                      │
│   ──────────────────────────                                                      │
│                                                                                   │
│   Step 0:  Start with K=4 copies of START token                                  │
│                                                                                   │
│            Beam 1: <START>  log_prob=0.0                                         │
│            Beam 2: <START>  log_prob=0.0                                         │
│            Beam 3: <START>  log_prob=0.0                                         │
│            Beam 4: <START>  log_prob=0.0                                         │
│                                                                                   │
│                                                                                   │
│   Step 1:  For each beam, compute next word probabilities                        │
│            Select top K across ALL continuations                                  │
│                                                                                   │
│            <START> → germany (0.40), the (0.30), after (0.15), ...             │
│                                                                                   │
│            Keep top 4:                                                            │
│            Beam 1: germany      log_prob=log(0.40)=-0.92                        │
│            Beam 2: the          log_prob=log(0.30)=-1.20                        │
│            Beam 3: after        log_prob=log(0.15)=-1.90                        │
│            Beam 4: a            log_prob=log(0.10)=-2.30                        │
│                                                                                   │
│                                                                                   │
│   Step 2:  Expand each beam, select top K overall                                │
│                                                                                   │
│            germany → won (0.35), defeated (0.30), beat (0.25), ...             │
│            the → game (0.40), match (0.30), ...                                 │
│            after → the (0.50), a (0.30), ...                                    │
│            a → great (0.40), good (0.35), ...                                   │
│                                                                                   │
│            All candidates with cumulative log probs:                             │
│            germany+won:       -0.92 + log(0.35) = -1.97                         │
│            germany+defeated:  -0.92 + log(0.30) = -2.12                         │
│            germany+beat:      -0.92 + log(0.25) = -2.31                         │
│            the+game:          -1.20 + log(0.40) = -2.12                         │
│            the+match:         -1.20 + log(0.30) = -2.40                         │
│            after+the:         -1.90 + log(0.50) = -2.59                         │
│            a+great:           -2.30 + log(0.40) = -3.22                         │
│            ...                                                                    │
│                                                                                   │
│            Keep top 4:                                                            │
│            Beam 1: germany won         log_prob=-1.97                           │
│            Beam 2: germany defeated    log_prob=-2.12                           │
│            Beam 3: the game            log_prob=-2.12                           │
│            Beam 4: germany beat        log_prob=-2.31                           │
│                                                                                   │
│   Continue until all beams end with STOP or reach max length.                    │
│   Return the beam with highest probability.                                       │
│                                                                                   │
│                                                                                   │
│   KEY INSIGHT:                                                                    │
│   ────────────                                                                    │
│                                                                                   │
│   Beam search explores more of the search space than greedy,                    │
│   but is still tractable (not exponential like exhaustive search).              │
│                                                                                   │
│   • Greedy: explores 1 path                                                      │
│   • Beam (K=4): explores ~4 paths                                               │
│   • Exhaustive: explores V^T paths (infeasible!)                                │
│                                                                                   │
│   where V = vocab size, T = sequence length                                     │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Hypothesis Class

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        HYPOTHESIS CLASS                                           │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   PURPOSE: Represent a single beam (partial sequence) during decoding            │
│                                                                                   │
│   ATTRIBUTES:                                                                     │
│   ───────────                                                                     │
│                                                                                   │
│   tokens      : List of token IDs generated so far                               │
│                 [START_ID, word1_id, word2_id, ...]                              │
│                                                                                   │
│   log_probs   : List of log probabilities for each token                         │
│                 [0.0, log(P(word1)), log(P(word2)), ...]                         │
│                                                                                   │
│   state       : Decoder hidden state after generating these tokens               │
│                 (cell_state, hidden_state) for LSTM                              │
│                                                                                   │
│   attn_dists  : List of attention distributions (one per step)                   │
│                 For visualization/analysis                                        │
│                                                                                   │
│   p_gens      : List of p_gen values (one per step)                              │
│                 For visualization/analysis                                        │
│                                                                                   │
│   coverage    : Coverage vector (if using coverage mechanism)                    │
│                                                                                   │
│                                                                                   │
│   KEY METHODS:                                                                    │
│   ────────────                                                                    │
│                                                                                   │
│   extend(token, log_prob, state, attn_dist, p_gen, coverage):                   │
│       Create new Hypothesis with one more token appended.                        │
│       Returns NEW Hypothesis (immutable design).                                 │
│                                                                                   │
│   latest_token:                                                                   │
│       Property that returns the last token (for next step input).               │
│                                                                                   │
│   log_prob:                                                                       │
│       Property that returns sum of all log probabilities.                        │
│       This is the sequence probability in log space.                             │
│                                                                                   │
│   avg_log_prob:                                                                   │
│       Property for length-normalized probability.                                │
│       log_prob / num_tokens (to avoid bias toward short sequences).             │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Implementation (beam_search.py)

```python
# beam_search.py: Hypothesis class (Lines 24-95)

class Hypothesis(object):
    """
    Represents a single beam hypothesis during beam search.
    """
    
    def __init__(self, tokens, log_probs, state, attn_dists, p_gens, coverage):
        """
        Initialize a hypothesis.
        
        Args:
            tokens: List of token IDs (including START)
            log_probs: List of log probabilities
            state: Decoder state tuple (c, h)
            attn_dists: List of attention distributions
            p_gens: List of p_gen values
            coverage: Current coverage vector
        """
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.attn_dists = attn_dists
        self.p_gens = p_gens
        self.coverage = coverage
    
    def extend(self, token, log_prob, state, attn_dist, p_gen, coverage):
        """
        Return NEW Hypothesis with token appended.
        (Immutable: doesn't modify self)
        """
        return Hypothesis(
            tokens=self.tokens + [token],
            log_probs=self.log_probs + [log_prob],
            state=state,
            attn_dists=self.attn_dists + [attn_dist],
            p_gens=self.p_gens + [p_gen],
            coverage=coverage
        )
    
    @property
    def latest_token(self):
        """Get the last token generated."""
        return self.tokens[-1]
    
    @property
    def log_prob(self):
        """Get total log probability of sequence."""
        return sum(self.log_probs)
    
    @property
    def avg_log_prob(self):
        """Get average log probability (length-normalized)."""
        return self.log_prob / len(self.tokens)
```

---

## Beam Search Algorithm

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                     BEAM SEARCH ALGORITHM                                         │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   ALGORITHM OVERVIEW:                                                             │
│   ────────────────────                                                            │
│                                                                                   │
│   1. ENCODE: Run encoder once to get encoder states                              │
│   2. INITIALIZE: Create K hypotheses with START token                            │
│   3. LOOP until done:                                                             │
│      a. Run decoder on all K hypotheses (batched)                                │
│      b. Get top-K × V candidate extensions                                       │
│      c. Keep top K overall                                                       │
│      d. Move completed hypotheses to results                                     │
│   4. RETURN: Best completed hypothesis                                           │
│                                                                                   │
│                                                                                   │
│   DETAILED ALGORITHM:                                                             │
│   ────────────────────                                                            │
│                                                                                   │
│   def run_beam_search(model, vocab, batch):                                      │
│       # Step 1: Encode                                                            │
│       enc_states, dec_init_state = model.run_encoder(batch)                      │
│                                                                                   │
│       # Step 2: Initialize                                                        │
│       hyps = [Hypothesis(                                                         │
│           tokens=[START_ID],                                                      │
│           log_probs=[0.0],                                                        │
│           state=dec_init_state,                                                  │
│           attn_dists=[],                                                         │
│           p_gens=[],                                                             │
│           coverage=zeros(enc_len)  # if using coverage                           │
│       ) for _ in range(beam_size)]                                               │
│                                                                                   │
│       results = []  # Completed hypotheses                                        │
│                                                                                   │
│       # Step 3: Loop                                                              │
│       for step in range(max_dec_steps):                                          │
│           # Get latest tokens from all hypotheses                                │
│           latest_tokens = [h.latest_token for h in hyps]                         │
│           # Get states from all hypotheses                                       │
│           states = [h.state for h in hyps]                                       │
│                                                                                   │
│           # Run decoder (batched over hypotheses)                                │
│           (topk_ids, topk_log_probs, new_states,                                │
│            attn_dists, p_gens, new_coverages) = model.decode_onestep(           │
│               latest_tokens, states, enc_states, ...)                           │
│                                                                                   │
│           # Generate all possible extensions                                      │
│           all_hyps = []                                                          │
│           for i, h in enumerate(hyps):                                           │
│               for j in range(2 * beam_size):  # top 2K continuations            │
│                   new_hyp = h.extend(                                            │
│                       token=topk_ids[i, j],                                      │
│                       log_prob=topk_log_probs[i, j],                            │
│                       state=new_states[i],                                       │
│                       attn_dist=attn_dists[i],                                   │
│                       p_gen=p_gens[i],                                           │
│                       coverage=new_coverages[i]                                  │
│                   )                                                               │
│                   all_hyps.append(new_hyp)                                       │
│                                                                                   │
│           # Sort by log probability and keep top K                               │
│           all_hyps = sorted(all_hyps, key=lambda h: h.log_prob,                 │
│                            reverse=True)                                         │
│                                                                                   │
│           # Separate completed from incomplete                                    │
│           hyps = []                                                               │
│           for h in all_hyps:                                                     │
│               if h.latest_token == STOP_ID:                                      │
│                   results.append(h)  # Completed!                                │
│               else:                                                               │
│                   hyps.append(h)                                                 │
│                                                                                   │
│               if len(hyps) == beam_size:                                         │
│                   break                                                          │
│                                                                                   │
│           # Stop if enough results                                                │
│           if len(results) >= beam_size:                                          │
│               break                                                              │
│                                                                                   │
│       # Step 4: Return best                                                       │
│       results = sorted(results, key=lambda h: h.avg_log_prob,                   │
│                       reverse=True)                                              │
│       return results[0]                                                          │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Details

### run_beam_search Function (beam_search.py)

```python
# beam_search.py: run_beam_search function (Lines 97-167)

def run_beam_search(sess, model, vocab, batch):
    """
    Run beam search decoding.
    
    Args:
        sess: TensorFlow session
        model: SummarizationModel
        vocab: Vocabulary
        batch: Batch object (single example for decoding)
    
    Returns:
        best_hyp: Hypothesis with best score
    """
    # Run encoder ONCE
    enc_states, dec_in_state = model.run_encoder(sess, batch)
    
    # Initialize hypotheses
    hyps = [Hypothesis(
        tokens=[vocab.word2id(vocab.START_DECODING)],
        log_probs=[0.0],
        state=dec_in_state,
        attn_dists=[],
        p_gens=[],
        coverage=np.zeros([batch.enc_batch.shape[1]])  # [enc_len]
    )]
    
    results = []  # Completed hypotheses
    steps = 0
    
    while steps < FLAGS.max_dec_steps and len(results) < FLAGS.beam_size:
        # Get latest tokens (batch together)
        latest_tokens = [h.latest_token for h in hyps]
        latest_tokens = [t if t < vocab.size() else vocab.word2id(vocab.UNKNOWN_TOKEN)
                        for t in latest_tokens]  # Replace OOV with UNK for embedding
        
        # Get states
        states = [h.state for h in hyps]
        
        # Get coverage vectors
        prev_coverage = [h.coverage for h in hyps]
        
        # Run decoder ONE STEP
        (topk_ids, topk_log_probs, 
         new_states, attn_dists, 
         p_gens, new_coverage) = model.decode_onestep(
            sess=sess,
            batch=batch,
            latest_tokens=latest_tokens,
            enc_states=enc_states,
            dec_init_states=states,
            prev_coverage=prev_coverage
        )
        
        # Extend hypotheses
        all_hyps = []
        num_orig_hyps = 1 if steps == 0 else len(hyps)
        
        for i in range(num_orig_hyps):
            h = hyps[i]
            state = new_states[i]
            attn_dist = attn_dists[i]
            p_gen = p_gens[i]
            coverage = new_coverage[i]
            
            for j in range(FLAGS.beam_size * 2):  # Consider top 2*K extensions
                new_hyp = h.extend(
                    token=topk_ids[i, j],
                    log_prob=topk_log_probs[i, j],
                    state=state,
                    attn_dist=attn_dist,
                    p_gen=p_gen,
                    coverage=coverage
                )
                all_hyps.append(new_hyp)
        
        # Sort and filter
        hyps = []
        for h in sort_hyps(all_hyps):
            if h.latest_token == vocab.word2id(vocab.STOP_DECODING):
                if steps >= FLAGS.min_dec_steps:  # Minimum length
                    results.append(h)
            else:
                hyps.append(h)
            if len(hyps) == FLAGS.beam_size or len(results) == FLAGS.beam_size:
                break
        
        steps += 1
    
    # If no complete results, use incomplete
    if len(results) == 0:
        results = hyps
    
    # Sort by length-normalized log probability
    results_sorted = sort_hyps(results)
    return results_sorted[0]


def sort_hyps(hyps):
    """Sort hypotheses by average log probability."""
    return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)
```

---

## Length Normalization

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                      LENGTH NORMALIZATION                                         │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   THE PROBLEM:                                                                    │
│   ────────────                                                                    │
│                                                                                   │
│   Log probability is CUMULATIVE (sum of log probs).                              │
│   Longer sequences have more terms → lower (more negative) scores.               │
│                                                                                   │
│   Example:                                                                        │
│   Short: "Germany won" → log_prob = -1.0 + -0.8 = -1.8                          │
│   Long:  "Germany defeated Argentina" → log_prob = -1.0 + -1.2 + -1.5 = -3.7   │
│                                                                                   │
│   Without normalization, beam search prefers SHORTER sequences!                  │
│   This leads to truncated outputs.                                               │
│                                                                                   │
│                                                                                   │
│   THE SOLUTION: Length Normalization                                              │
│   ─────────────────────────────────────                                           │
│                                                                                   │
│   Divide by sequence length:                                                      │
│                                                                                   │
│   score = log_prob / length                                                      │
│                                                                                   │
│   Or equivalently, use average log probability:                                  │
│                                                                                   │
│   score = (1/T) × Σ log P(y_t | y_<t, x)                                        │
│                                                                                   │
│   Example after normalization:                                                    │
│   Short: "Germany won" → avg_log_prob = -1.8 / 2 = -0.90                        │
│   Long:  "Germany defeated Argentina" → avg_log_prob = -3.7 / 3 = -1.23        │
│                                                                                   │
│   Now shorter sequence wins (higher is better for log prob)!                    │
│   ...but this is correct because short sequence has better per-word quality.    │
│                                                                                   │
│                                                                                   │
│   IMPLEMENTATION:                                                                 │
│   ───────────────                                                                 │
│                                                                                   │
│   class Hypothesis:                                                               │
│       @property                                                                   │
│       def log_prob(self):                                                        │
│           """Total log probability."""                                           │
│           return sum(self.log_probs)                                             │
│                                                                                   │
│       @property                                                                   │
│       def avg_log_prob(self):                                                    │
│           """Length-normalized log probability."""                               │
│           return self.log_prob / len(self.tokens)                               │
│                                                                                   │
│   # During beam search, sort by avg_log_prob                                     │
│   results_sorted = sorted(results, key=lambda h: h.avg_log_prob,                │
│                          reverse=True)                                           │
│                                                                                   │
│                                                                                   │
│   VARIATIONS:                                                                     │
│   ───────────                                                                     │
│                                                                                   │
│   1. Simple normalization: score = log_prob / length                            │
│                                                                                   │
│   2. Length penalty (Google NMT): score = log_prob / length^α                   │
│      where α ∈ [0, 1] controls the penalty strength                             │
│      α = 0: no normalization                                                     │
│      α = 1: standard normalization                                               │
│      α = 0.6-0.8: commonly used in practice                                     │
│                                                                                   │
│   3. Coverage penalty: Add bonus for covering source                            │
│      score = log_prob / length + β × coverage_bonus                             │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Stopping Criteria

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                       STOPPING CRITERIA                                           │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   WHEN DOES BEAM SEARCH STOP?                                                     │
│   ───────────────────────────                                                     │
│                                                                                   │
│   1. HYPOTHESIS COMPLETION                                                        │
│      ──────────────────────                                                       │
│      When a hypothesis generates STOP token:                                      │
│      • Move to results list                                                       │
│      • Remove from active beams                                                   │
│      • Continue with remaining beams                                              │
│                                                                                   │
│      if h.latest_token == STOP_ID:                                               │
│          results.append(h)  # Completed!                                         │
│      else:                                                                        │
│          hyps.append(h)     # Still searching                                    │
│                                                                                   │
│                                                                                   │
│   2. MAXIMUM STEPS REACHED                                                        │
│      ──────────────────────                                                       │
│      Prevent infinite decoding:                                                   │
│                                                                                   │
│      while steps < max_dec_steps:                                                │
│          ...                                                                      │
│                                                                                   │
│      Default: max_dec_steps = 100                                                │
│                                                                                   │
│                                                                                   │
│   3. MINIMUM STEPS (for completed)                                                │
│      ─────────────────────────────                                                │
│      Prevent very short outputs:                                                  │
│                                                                                   │
│      if h.latest_token == STOP_ID:                                               │
│          if steps >= min_dec_steps:                                              │
│              results.append(h)                                                   │
│          # else: discard this hypothesis                                         │
│                                                                                   │
│      Default: min_dec_steps = 35                                                 │
│                                                                                   │
│                                                                                   │
│   4. ENOUGH RESULTS                                                               │
│      ──────────────                                                               │
│      Stop when we have beam_size completed hypotheses:                           │
│                                                                                   │
│      if len(results) >= beam_size:                                               │
│          break                                                                   │
│                                                                                   │
│      Why? We have enough candidates to choose from.                              │
│                                                                                   │
│                                                                                   │
│   FALLBACK: NO COMPLETE RESULTS                                                   │
│   ────────────────────────────────                                                │
│                                                                                   │
│   If no hypothesis completed (all reached max steps without STOP):               │
│                                                                                   │
│   if len(results) == 0:                                                          │
│       results = hyps  # Use incomplete hypotheses                                │
│                                                                                   │
│   This ensures we always return something!                                       │
│                                                                                   │
│                                                                                   │
│   HYPERPARAMETERS:                                                                │
│   ─────────────────                                                               │
│                                                                                   │
│   Parameter        Default    Description                                        │
│   ─────────        ───────    ───────────                                        │
│   beam_size        4          Number of beams to keep                            │
│   max_dec_steps    100        Maximum decoding length                            │
│   min_dec_steps    35         Minimum length before accepting STOP               │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Worked Example

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   COMPLETE BEAM SEARCH EXAMPLE                                    │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   SETUP:                                                                          │
│   ──────                                                                          │
│   • beam_size = 2 (for simplicity)                                               │
│   • vocab = {<s>=0, </s>=1, germany=2, won=3, defeated=4, the=5,                │
│              cup=6, argentina=7, world=8}                                        │
│   • Source: "Germany beat Argentina in the World Cup"                            │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   STEP 0: INITIALIZE                                                              │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   Hyp 1: tokens=[<s>], log_prob=0.0, avg_log_prob=0.0                           │
│                                                                                   │
│   (Only 1 hypothesis at start)                                                   │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   STEP 1: FIRST WORD                                                              │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   Run decoder with input <s>:                                                    │
│                                                                                   │
│   P(germany)   = 0.50  → log_prob = -0.69                                       │
│   P(the)       = 0.25  → log_prob = -1.39                                       │
│   P(argentina) = 0.15  → log_prob = -1.90                                       │
│   P(won)       = 0.05  → log_prob = -3.00                                       │
│   ...                                                                             │
│                                                                                   │
│   Keep top 2:                                                                     │
│   Hyp 1: [<s>, germany]    log_prob=-0.69   avg=-0.35                           │
│   Hyp 2: [<s>, the]        log_prob=-1.39   avg=-0.70                           │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   STEP 2: SECOND WORD                                                             │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   Run decoder on both hypotheses:                                                │
│                                                                                   │
│   From "germany":                                                                 │
│   P(won)      = 0.40  → cumulative: -0.69 + -0.92 = -1.61                       │
│   P(defeated) = 0.35  → cumulative: -0.69 + -1.05 = -1.74                       │
│   P(beat)     = 0.15  → cumulative: -0.69 + -1.90 = -2.59                       │
│                                                                                   │
│   From "the":                                                                     │
│   P(world)    = 0.30  → cumulative: -1.39 + -1.20 = -2.59                       │
│   P(cup)      = 0.25  → cumulative: -1.39 + -1.39 = -2.78                       │
│                                                                                   │
│   All candidates ranked by cumulative log_prob:                                  │
│   1. germany won        -1.61   avg=-0.54                                       │
│   2. germany defeated   -1.74   avg=-0.58                                       │
│   3. germany beat       -2.59   avg=-0.86                                       │
│   4. the world          -2.59   avg=-0.86                                       │
│   5. the cup            -2.78   avg=-0.93                                       │
│                                                                                   │
│   Keep top 2:                                                                     │
│   Hyp 1: [<s>, germany, won]       log_prob=-1.61   avg=-0.54                   │
│   Hyp 2: [<s>, germany, defeated]  log_prob=-1.74   avg=-0.58                   │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   STEP 3: THIRD WORD                                                              │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   From "germany won":                                                             │
│   P(the)      = 0.50  → cumulative: -1.61 + -0.69 = -2.30                       │
│   P(</s>)     = 0.10  → cumulative: -1.61 + -2.30 = -3.91  [COMPLETE!]         │
│                                                                                   │
│   From "germany defeated":                                                        │
│   P(argentina)= 0.60  → cumulative: -1.74 + -0.51 = -2.25  ← Best!             │
│   P(the)      = 0.20  → cumulative: -1.74 + -1.61 = -3.35                       │
│                                                                                   │
│   Candidates:                                                                     │
│   1. germany defeated argentina  -2.25  avg=-0.56                               │
│   2. germany won the             -2.30  avg=-0.58                               │
│   3. germany won </s>            -3.91  avg=-0.98  [COMPLETE]                   │
│   4. germany defeated the        -3.35  avg=-0.84                               │
│                                                                                   │
│   Keep top 2 active:                                                              │
│   Hyp 1: [<s>, germany, defeated, argentina]  log_prob=-2.25                    │
│   Hyp 2: [<s>, germany, won, the]             log_prob=-2.30                    │
│                                                                                   │
│   Results (completed):                                                            │
│   Result 1: "germany won </s>" log_prob=-3.91 (if min_dec_steps satisfied)      │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   CONTINUE UNTIL...                                                               │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   • 2 hypotheses complete, OR                                                     │
│   • max_dec_steps reached                                                         │
│                                                                                   │
│   Let's say we continue and get:                                                  │
│                                                                                   │
│   Results (completed):                                                            │
│   1. "germany defeated argentina to win the cup </s>"                           │
│      log_prob=-5.50  avg=-0.61                                                  │
│   2. "germany won the world cup </s>"                                           │
│      log_prob=-4.20  avg=-0.60                                                  │
│                                                                                   │
│   FINAL OUTPUT (best by avg_log_prob):                                           │
│   "germany won the world cup"                                                    │
│                                                                                   │
│   (avg=-0.60 > avg=-0.61)                                                        │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

Beam Search key concepts:

1. **Keeps top-K hypotheses** at each step instead of just 1 (greedy)
2. **Explores more paths** in the search space
3. **Hypothesis class** tracks tokens, probabilities, and states
4. **Length normalization** prevents bias toward shorter sequences
5. **Stopping criteria**: STOP token, max steps, enough results
6. **Returns best hypothesis** by average log probability

The pointer-generator uses beam_size=4 by default, which provides a good balance between search quality and computation cost.

---

*Next: [10_training_pipeline.md](10_training_pipeline.md) - Training Loop and Optimization*
