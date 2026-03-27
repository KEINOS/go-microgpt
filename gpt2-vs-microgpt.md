# GPT-2 Core and microgpt Simplifications

This note compares canonical GPT-2 with the microgpt implementation used in this repository, based on:

1. `ref/microgpt.py` (Python reference)
2. `microgpt.go` (Go port in this repository)
3. `README.md` (project-level architecture notes)

The goal is to separate what stays structurally GPT-2-like from what is intentionally simplified for learning.

## Scope and baseline

The baseline in this note is the decoder-only Transformer path:

1. Token + position embeddings are summed.
2. A stack of Transformer blocks is applied.
3. Each block performs attention and MLP sublayers with residual connections.
4. Hidden states are projected to vocabulary logits.
5. Softmax is applied outside the core forward path (loss/sampling stage).

In pre-norm form:

```go
x = x + SelfAttention(Norm(x))
x = x + MLP(Norm(x))
```

## Canonical GPT-2 (reference shape)

Canonical GPT-2 uses LayerNorm in blocks, GELU in MLP, and a final LayerNorm before `lm_head`.

Abbreviations:

- `wte`: token embedding table
- `wpe`: positional embedding table
- `lm_head`: projection from hidden state to vocabulary logits

```mermaid
flowchart TB
    IN([Input Token IDs]) --> WTE[Token Embedding\nwte]
    IN --> WPE[Position Embedding\nwpe]
    WTE --> ADD0((Add))
    WPE --> ADD0

    ADD0 --> BLK

    subgraph BLK[Transformer Block x N]
        direction TB
        LN1[LayerNorm] --> ATTN[Causal Multi-Head Self-Attention]
        ATTN --> ADD1((Add Residual))
        ADD1 --> LN2[LayerNorm]
        LN2 --> MLP[MLP: Linear -> GELU -> Linear]
        MLP --> ADD2((Add Residual))
    end

    ADD2 --> FLN[Final LayerNorm]
    FLN --> HEAD[Linear Projection\nlm_head]
    HEAD --> LOGITS([Logits])
    LOGITS --> SMX[Softmax when needed]
    SMX --> OUT([Token Probabilities])
```

## microgpt in this repository (Python + Go)

Both `ref/microgpt.py` and `microgpt.go` keep the same high-level flow and simplify internals.

```mermaid
flowchart TB
    IN([Input Token IDs]) --> WTE[Token Embedding\nwte]
    IN --> WPE[Position Embedding\nwpe]
    WTE --> ADD0((Add))
    WPE --> ADD0
    ADD0 --> N0[RMSNorm after embedding sum]

    N0 --> BLK

    subgraph BLK[Transformer Block x N]
        direction TB
        N1[RMSNorm] --> ATTN[Causal MHA via prefix KV cache<br/>no explicit mask tensor]
        ATTN --> ADD1((Add Residual))
        ADD1 --> N2[RMSNorm]
        N2 --> MLP[MLP: Linear -> ReLU -> Linear]
        MLP --> ADD2((Add Residual))
    end

    ADD2 --> HEAD[Linear Projection\nlm_head]
    HEAD --> LOGITS([Logits])
    LOGITS --> SMX[Softmax when needed]
    SMX --> OUT([Token Probabilities])
```

## Simplifications used here

### 1) LayerNorm -> RMSNorm

RMSNorm is used instead of LayerNorm. It avoids mean-centering and is easier to express in a scalar autograd graph.

Important implementation detail:

- In `ref/microgpt.py` and `microgpt.go`, RMSNorm is parameter-free (no learnable gamma scale parameter), by design.

### 2) GELU -> ReLU

The MLP activation is ReLU instead of GELU. This simplifies forward/backward computation while keeping block topology unchanged.

### 3) Bias terms removed

Linear layers are implemented without bias vectors.

### 4) Extra norm after embedding sum

An RMSNorm is applied immediately after `wte + wpe`, before entering the first block.

### 5) No final norm before `lm_head`

Unlike canonical GPT-2, this implementation projects to logits directly after the last block.

### 6) Causality via autoregressive prefix, not explicit mask tensor

This implementation enforces causality by incremental prefix K/V accumulation (no explicit mask tensor). As a trade-off, it currently uses token-by-token forward passes instead of full-sequence parallel attention with masking.

#### Causal Masking in Attention

Both canonical GPT-2 and microgpt share the same core scaled dot-product attention computation.

The key difference lies in how causality is enforced:

- **Canonical GPT-2**:
  - Uses an **explicit causal mask** (lower triangular matrix).
  - Full sequences are processed in parallel during training, with future tokens masked by setting their attention scores to `-∞` before softmax.
- **microgpt (this implementation)**:
  - Does **not** use an explicit mask tensor at all.
  - Instead, causality is enforced implicitly through **incremental K/V accumulation** in a prefix cache - only past keys and values are stored and visible to the model.
  - Since tokens are processed one-by-one during both training and inference, future tokens are never fed into the attention mechanism.

This design significantly simplifies the attention code and makes the forward pass easier to follow, though it relies on token-by-token execution rather than batched full-sequence attention with masking.

### 7) No dropout regularization

Dropout is intentionally omitted for simplicity.

### 8) Separate `wte` and `lm_head` weights

`microgpt.go` keeps `wte` and `lm_head` as separate matrices to stay faithful to the Python reference in this repository.

## Implementation notes for the Go port

From `microgpt.go` and `README.md`:

1. The architecture is a direct educational port, prioritizing readability and structural faithfulness over efficiency.
2. Training and inference are both autoregressive, using token-by-token forward calls with incremental K/V accumulation.
3. Softmax remains outside the core block pipeline (cross-entropy in training, temperature sampling in inference).

## Quick comparison table

| Aspect | Canonical GPT-2 | microgpt in this repository |
| :--- | :--- | :--- |
| Model family | Decoder-only Transformer | Decoder-only Transformer |
| Block structure | Attention + MLP with residuals | Attention + MLP with residuals |
| Normalization type | LayerNorm | RMSNorm |
| Learnable norm scale | Yes (LayerNorm gamma/beta) | No (parameter-free RMSNorm, no learnable gamma scale parameter) |
| Norm after embedding sum | No | Yes |
| MLP activation | GELU | ReLU |
| Dropout | Used in training | Omitted |
| Linear bias terms | Used | Omitted |
| Final norm before `lm_head` | Yes | No |
| Causal masking | Explicit causal masking in attention logic | No explicit mask tensor (incremental prefix K/V cache) |
| `wte`/`lm_head` weight tying | Implementation-dependent across GPT-2 codebases | Not tied (separate matrices) |
| Output of forward pass | Logits | Logits |
| Softmax usage | Outside core path | Outside core path |

## Closing note

The structural takeaway is:

1. Keep the GPT-2 decoder-only execution path.
2. Apply simplifications that reduce autograd and implementation complexity.

With this boundary, `ref/microgpt.py` and `microgpt.go` are compact, inspectable maps of GPT-style modeling logic for learning.
