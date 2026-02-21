# go-microgpt Specification

Port of `ref/microgpt.py` (201 lines, Python stdlib only) to a single Go file `microgpt.go`.
All behavior must match the Python reference exactly.

## 1. Overview

The program:

1. Downloads a names dataset (`input.txt`) on first run.
2. Builds a character-level vocabulary from all unique characters.
3. Trains a small GPT model for 1000 steps using the Adam optimizer.
4. Prints 20 sampled names after training.

## 2. Configuration & Constants

| Name           | Value    | Description                         |
| :------------- | :------- | :---------------------------------- |
| `nLayer`       | 1        | Number of transformer layers        |
| `nEmbd`        | 16       | Embedding dimension                 |
| `blockSize`    | 16       | Maximum context window length       |
| `nHead`        | 4        | Number of attention heads           |
| `headDim`      | 4        | `nEmbd / nHead`, dimension per head |
| `numSteps`     | 1000     | Training steps                      |
| `learningRate` | 0.01     | Initial learning rate               |
| `beta1`        | 0.85     | Adam first-moment decay             |
| `beta2`        | 0.99     | Adam second-moment decay            |
| `epsAdam`      | 1e-8     | Adam epsilon                        |
| `temperature`  | 0.5      | Sampling temperature for inference  |
| `initStd`      | 0.08     | Std dev for weight initialization   |
| `randomSeed`   | 42       | Global RNG seed                     |
| `dataURL`      | see code | Dataset URL from karpathy/makemore  |

## 3. Data Structures

### `Value` — scalar autograd node

```go
type Value struct {
    Data       float64
    Grad       float64
    children   []*Value
    localGrads []float64
}
```

Every scalar in the computation is a `*Value`. Pointer semantics are required so that
shared weight nodes accumulate gradients correctly via `+=`.

### Model parameters

```go
// state_dict equivalent
type StateDict map[string][][]*Value

// Flat list of all parameters for the optimizer
var params []*Value
```

### KV cache (per forward pass)

```go
// shape: [nLayer][timeStep][nEmbd]
keys   [][][]*Value
values [][][]*Value
```

## 4. Dataset Loading & Tokenization

**Loading (`input.txt`):**

- If `input.txt` does not exist, download from `dataURL` using `net/http`.
- Read all lines, strip whitespace, discard empty lines → `docs []string`.
- Shuffle `docs` in-place using the seeded RNG.

**Vocabulary:**

- Collect all unique characters across `docs`.
- Sort them → `uchars []rune` (sorted unique characters).
- `BOS = len(uchars)` — the beginning/end-of-sequence token ID.
- `vocabSize = len(uchars) + 1`.

For the names dataset: 26 lowercase letters, so `BOS=26`, `vocabSize=27`.

**Encoding:** index of character in `uchars`.
**Decoding:** `uchars[tokenID]` (only for non-BOS tokens).

## 5. Autograd: `Value`

### Constructor

```go
func newValue(data float64, children []*Value, localGrads []float64) *Value
```

Initializes `Grad = 0`.

### Operations

Each operation creates a new `*Value` node and stores its children + local gradients.

| Method / Function | Forward `out.Data`           | `localGrads` (one per child in order) |
| :---------------- | :--------------------------- | :------------------------------------ |
| `add(a, b)`       | `a.Data + b.Data`            | `[1, 1]`                              |
| `mul(a, b)`       | `a.Data * b.Data`            | `[b.Data, a.Data]`                    |
| `pow(a, exp)`     | `a.Data ^ exp` (exp: float64)| `[exp * a.Data^(exp-1)]`              |
| `(v).log()`       | `math.Log(v.Data)`           | `[1 / v.Data]`                        |
| `(v).exp()`       | `math.Exp(v.Data)`           | `[math.Exp(v.Data)]`                  |
| `(v).relu()`      | `max(0, v.Data)`             | `[1 if v.Data > 0 else 0]`            |

Convenience wrappers (no new node, delegate to above):

- `neg(a)` → `mul(a, newValue(-1, ...))`
- `sub(a, b)` → `add(a, neg(b))`
- `div(a, b)` → `mul(a, pow(b, -1))`

### `backward()` — Backpropagation

Called on the root loss `*Value`. Algorithm:

1. Build a topological order of all reachable nodes via DFS:
   - Use `visited map[*Value]bool` to avoid revisiting.
   - Append each node **after** visiting its children (post-order).
2. Set `root.Grad = 1.0`.
3. Iterate over the topological list **in reverse** (root → leaves).
4. For each node `v`, propagate gradients to its children:

```go
child.Grad += localGrad[i] * v.Grad   // chain rule
```

The `+=` handles nodes that appear as children of multiple parents (shared weights).

## 6. Model Parameters (`state_dict`)

All matrices are initialized with `Normal(0, initStd)` (`initStd=0.08`).
**No bias vectors anywhere.**

| Key                   | Shape                | Description                  |
| :-------------------- | :------------------- | :--------------------------- |
| `wte`                 | `[vocabSize, nEmbd]` | Token embedding table        |
| `wpe`                 | `[blockSize, nEmbd]` | Position embedding table     |
| `lm_head`             | `[vocabSize, nEmbd]` | Output projection            |
| `layer{i}.attn_wq`    | `[nEmbd, nEmbd]`     | Attention query projection   |
| `layer{i}.attn_wk`    | `[nEmbd, nEmbd]`     | Attention key projection     |
| `layer{i}.attn_wv`    | `[nEmbd, nEmbd]`     | Attention value projection   |
| `layer{i}.attn_wo`    | `[nEmbd, nEmbd]`     | Attention output projection  |
| `layer{i}.mlp_fc1`    | `[4*nEmbd, nEmbd]`   | MLP expand layer             |
| `layer{i}.mlp_fc2`    | `[nEmbd, 4*nEmbd]`   | MLP contract layer           |

`params` is all matrices flattened: `for each matrix → for each row → for each element`.

With defaults: 4192 total parameters.

## 7. Helper Functions

### `linear(x []*Value, w [][]*Value) []*Value`

Matrix-vector product `y = W @ x`. No bias.

```go
y[i] = sum(w[i][j] * x[j]  for j in 0..len(x))
```

Returns a slice of length `len(w)`.

### `softmax(logits []*Value) []*Value`

Numerically stable softmax:

```go
maxVal = max(v.Data for v in logits)   // plain float, NOT a Value
exps[i] = (logits[i] - maxVal).exp()  // logits[i] is a Value; maxVal is a plain float subtracted
total = sum(exps)
return [e / total for e in exps]
```

`maxVal` must be extracted as a plain `float64` — it must **not** become a `Value` node
(it is a numerical shift, not part of the computation graph).

### `rmsnorm(x []*Value) []*Value`

```go
ms    = sum(xi * xi for xi in x) / len(x)   // Value
scale = pow(ms + 1e-5, -0.5)                // Value
return [xi * scale for xi in x]
```

No learnable scale parameter. Epsilon = `1e-5`.

## 8. GPT Forward Pass

```go
func gpt(tokenID, posID int, keys, values [][][]*Value, stateDict StateDict) []*Value
```

Returns `logits []*Value` of length `vocabSize`.

### Step 1 — Input embedding

```go
tokEmb = stateDict["wte"][tokenID]      // []*Value, len=nEmbd
posEmb = stateDict["wpe"][posID]        // []*Value, len=nEmbd
x = tokEmb[i] + posEmb[i]  for each i  // element-wise add
x = rmsnorm(x)
```

### Step 2 — Transformer layers (repeat `nLayer` times, index `li`)

**Attention sub-block:**

```go
xRes = x
x = rmsnorm(x)
q = linear(x, stateDict["layer{li}.attn_wq"])  // len=nEmbd
k = linear(x, stateDict["layer{li}.attn_wk"])  // len=nEmbd
v = linear(x, stateDict["layer{li}.attn_wv"])  // len=nEmbd
keys[li]   = append(keys[li], k)
values[li] = append(values[li], v)
```

Per-head attention (for each head `h` in `0..nHead`):

```go
hs   = h * headDim
q_h  = q[hs : hs+headDim]
k_h  = [ki[hs : hs+headDim] for each past ki in keys[li]]
v_h  = [vi[hs : hs+headDim] for each past vi in values[li]]

// Scaled dot-product scores
score[t] = sum(q_h[j] * k_h[t][j]) / sqrt(headDim)   for each past t

// Attention weights
attnW = softmax(score)

// Weighted value aggregation
headOut[j] = sum(attnW[t] * v_h[t][j])   for each t

xAttn = concat(headOut for all heads)   // len=nEmbd
```

Output projection + residual:

```go
x = linear(xAttn, stateDict["layer{li}.attn_wo"])
x = x[i] + xRes[i]  for each i
```

**MLP sub-block:**

```go
xRes = x
x = rmsnorm(x)
x = linear(x, stateDict["layer{li}.mlp_fc1"])  // expand: nEmbd -> 4*nEmbd
x = relu(xi) for each xi                        // element-wise ReLU
x = linear(x, stateDict["layer{li}.mlp_fc2"])  // contract: 4*nEmbd -> nEmbd
x = x[i] + xRes[i]  for each i
```

### Step 3 — Output logits

```go
logits = linear(x, stateDict["lm_head"])   // len=vocabSize
return logits
```

## 9. Training Loop

```go
for step = 0; step < numSteps; step++:
    // 1. Select document (round-robin)
    doc    = docs[step % len(docs)]
    tokens = [BOS] + encode(doc) + [BOS]
    n      = min(blockSize, len(tokens)-1)

    // 2. Forward pass
    keys, values = empty [nLayer][][] per layer
    losses = []
    for posID = 0; posID < n; posID++:
        tokenID = tokens[posID]
        targetID = tokens[posID+1]
        logits = gpt(tokenID, posID, keys, values, stateDict)
        probs  = softmax(logits)
        loss_t = neg(probs[targetID].log())
        losses = append(losses, loss_t)
    loss = (1/n) * sum(losses)
    // Go: use (1.0/float64(n))*sum(losses) — avoid integer division

    // 3. Backward pass
    loss.backward()

    // 4. Adam update
    lrT = learningRate * (1 - step/numSteps)
    // Go: use float64(step)/float64(numSteps) — avoid integer division
    for i, p in enumerate(params):
        m[i] = beta1*m[i] + (1-beta1)*p.Grad
        v[i] = beta2*v[i] + (1-beta2)*p.Grad^2
        mHat = m[i] / (1 - beta1^(step+1))
        vHat = v[i] / (1 - beta2^(step+1))
        p.Data -= lrT * mHat / (sqrt(vHat) + epsAdam)
        p.Grad = 0

    // 5. Log progress (overwrite same line)
    print("step {step+1} / {numSteps} | loss {loss.Data:.4f}")
```

Adam state vectors `m` and `v` are `[]float64`, initialized to all zeros, indexed
identically to `params`.

## 10. Inference

After training, generate 20 samples:

```go
for sampleIdx = 0; sampleIdx < 20; sampleIdx++:
    keys, values = empty [nLayer][][] per layer
    tokenID = BOS
    sample  = []rune{}
    for posID = 0; posID < blockSize; posID++:
        logits = gpt(tokenID, posID, keys, values, stateDict)
        // temperature scaling: divide each logit by temperature
        scaled = [l / temperature for l in logits]   // Value / float64
        probs  = softmax(scaled)
        // weighted random sample using probs[i].Data as weights
        tokenID = weightedChoice(range(vocabSize), weights=[p.Data for p in probs])
        if tokenID == BOS:
            break
        sample = append(sample, uchars[tokenID])
    print("sample {sampleIdx+1:2d}: {string(sample)}")
```

`weightedChoice` maps to `rand.New(source).Float64()` with a cumulative distribution
or Go's equivalent of Python's `random.choices`.

## 11. Go-Specific Implementation Notes

| Concern | Go approach |
| :------ | :---------- |
| Pointer semantics | Use `*Value` everywhere; shared weight nodes must accumulate `Grad` via `+=` |
| Topological sort visited set | `map[*Value]bool` (Go has no built-in set type) |
| RNG | `rand.New(rand.NewSource(42))` — one global seeded source |
| Dataset download | `net/http` GET, write to `input.txt` |
| `math.Log`, `math.Exp`, `math.Sqrt` | From Go stdlib `math` package |
| Plain-float operations in `softmax` | Extract `.Data` before use; do not create a `Value` node for `maxVal` |
| `vocabSize` and `BOS` | Computed at runtime from the dataset |
| Single file | All code in `microgpt.go` |
| No external dependencies | Only Go stdlib |
