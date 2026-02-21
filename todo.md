# go-microgpt Implementation Task List

TDD workflow: stub → test (fail) → implement → test (pass) → lint → edge cases.

Reference: `spec.md` for all function signatures and behavior.

---

## 0. Project Initialization & Setup

- [ ] Create `microgpt_test.go` file with package declaration
- [ ] Run `go mod tidy` to update go.mod and go.sum
- [ ] Create `.golangci.yml` configuration file for linting (referenced in CLAUDE.md but missing)
- [ ] Add file header comment to `microgpt.go` explaining the project origin (port of Karpathy's Python implementation)
- [ ] Verify `go build .` and `go test ./...` commands work (should fail gracefully with placeholder main)

---

## 0.1. ⚠️ Critical Go-Specific Warnings

**Read this section carefully before starting implementation. These are common pitfalls that will cause subtle bugs if not addressed.**

### Integer Division Trap

Go's `/` operator performs **integer division** when both operands are integers, which will produce incorrect results in floating-point calculations.

**Critical locations:**

- **Adam optimizer learning rate decay**: `lr_t = lr * (1 - step/numSteps)` → WRONG
  - ✅ Correct: `lr_t = lr * (1.0 - float64(step)/float64(numSteps))`
- **Training loss averaging**: `loss = (1/n) * sum(losses)` → WRONG
  - ✅ Correct: `loss = (1.0 / float64(n)) * sum(losses)`

**Test verification**: Create explicit unit tests checking that learning rate decays smoothly (not in discrete jumps) and loss calculation produces non-integer results.

### Pointer Semantics for Gradient Accumulation

All `Value` nodes **must be pointers** (`*Value`), not values. Shared weight nodes (used multiple times in computation graph) must accumulate gradients via `+=` during backward pass.

**Why pointers are required:**

- When a weight appears in multiple operations (e.g., matrix used for multiple inputs), the same `*Value` node is referenced multiple times
- During backward pass, each reference contributes its gradient: `child.Grad += localGrad * v.Grad`
- If `Value` were a struct (not pointer), each reference would be a separate copy, and gradients would not accumulate correctly

**Test verification**: Create `TestBackwardSharedNode` that uses the same `Value` twice in computation and verifies gradient is sum of both paths.

### Race Conditions in Backward Pass

The backward pass mutates shared `*Value` nodes (specifically `Grad` field) during gradient accumulation. Without proper testing, this can cause data races.

**Critical test command:**

```bash
go test --race ./...
go run --race .
```

**When to test:**

- After implementing `backward()` method (Section 4)
- After implementing full GPT forward/backward pass (Section 10)
- After implementing full training loop (Section 12)

**Expected behavior**: No race detector warnings. All gradient accumulation is sequential in backward pass (no goroutines), so races indicate a bug.

### Softmax Numerical Stability

The `softmax` function requires special handling of `maxVal` to prevent overflow.

**Critical requirement:** `maxVal` must be extracted as a **plain `float64`**, NOT a `*Value` node.

```go
// ❌ WRONG - creates unnecessary Value node in computation graph
maxVal := logits[0]
for _, v := range logits[1:] {
    if v.Data > maxVal.Data {
        maxVal = v  // still a *Value
    }
}

// ✅ CORRECT - extract as plain float64
maxVal := logits[0].Data  // float64, not *Value
for _, v := range logits[1:] {
    if v.Data > maxVal {
        maxVal = v.Data
    }
}
// Then: exps[i] = (logits[i] - maxVal).exp()  // subtract float64 from *Value
```

**Why:** `maxVal` is a numerical stability shift, not part of the differentiable computation. Creating a Value node for it would pollute the computation graph and affect gradients incorrectly.

**Test verification**: `TestSoftmaxNumericalStability` with very large logits (e.g., 1000.0) should not produce NaN or Inf.

---

## 0.5. Global Configuration & Constants

- [ ] Create stub for all global configuration constants in `microgpt.go`:
  - Model architecture: `nLayer=1`, `nEmbd=16`, `blockSize=16`, `nHead=4`, `headDim=4`
  - Training: `numSteps=1000`, `learningRate=0.01`, `beta1=0.85`, `beta2=0.99`, `epsAdam=1e-8`
  - Data: `randomSeed=42`, `dataURL` (names.txt from karpathy/makemore)
  - Inference: `temperature=0.5`, `initStd=0.08`
- [ ] Create test `TestConfigConstants` (verify all constants have expected values)
- [ ] Run `go test ./... -v` and confirm test passes (no fail step needed — pure constants always compile and pass)
- [ ] Add comments explaining each configuration parameter's purpose

### 0.5.A. Globals vs Parameters Clarification

- [ ] Document which state should be global vs passed as parameters:
  - **Global Constants (initialized once, never change):**
    - Model config: `nLayer`, `nEmbd`, `blockSize`, `nHead`, `headDim`
    - Training config: `numSteps`, `learningRate`, `beta1`, `beta2`, `epsAdam`
    - Data config: `randomSeed`, `dataURL`, `temperature`, `initStd`
    - Global RNG: `rng *rand.Rand`
  - **Parameters (passed explicitly to functions):**
    - `stateDict StateDict` - all model weights
    - `params []*Value` - flattened parameters for optimizer
    - `uchars []rune` - vocabulary mapping
    - `BOS int` - special token value
    - `vocabSize int` - total vocabulary size
    - `docs []string` - training documents
    - `optimizer *adamOptimizer` - optimizer state
  - **Rationale:** Passing state explicitly makes functions testable without side effects and avoids race conditions

---

## 0.7. Random Number Generator Initialization

- [ ] Create global RNG variable stub: `var rng *rand.Rand`
- [ ] Create function `initRNG(seed int64)` to initialize seeded RNG
- [ ] Create test `TestRNGSeeded` (verify same seed produces same sequence)
- [ ] Run `go test ./... -v` and confirm test fails
- [ ] Implement `initRNG` using `rand.New(rand.NewSource(seed))`
- [ ] Run `go test ./... -v` and confirm test passes
- [ ] Run `golangci-lint run --fix` and fix lint errors
- [ ] Add test for deterministic behavior across multiple runs with same seed

---

## 1. Value Type & Constructor

- [ ] Create `Value` struct stub in `microgpt.go` with fields: `Data`, `Grad`, `children`, `localGrads`
- [ ] Create test function `TestNewValue` in `microgpt_test.go` (initialize a Value, check fields)
- [ ] Run `go test ./... -v` and confirm test fails
- [ ] Implement `newValue(data float64, children []*Value, localGrads []float64) *Value`
- [ ] Run `go test ./... -v` and confirm test passes
- [ ] Run `golangci-lint run --fix` and fix lint errors
- [ ] Add edge case tests: zero value, negative data, no children

---

## 1.A. Type Definitions

- [ ] Create `StateDict` type definition: `type StateDict map[string][][]*Value`
- [ ] Add documentation comment explaining StateDict stores all model weight matrices
- [ ] No test needed (pure type definition)

---

## 2. Value Arithmetic Operations (add, mul, pow, neg, sub, div)

- [ ] Create function stubs for `add`, `mul`, `pow`, `neg`, `sub`, `div` (all return `*Value`)
- [ ] Create test functions `TestValueAdd`, `TestValueMul`, `TestValuePow`, `TestValueNeg`, `TestValueSub`, `TestValueDiv` (verify forward computation and children/localGrads storage; verify neg/sub/div delegate to core ops)
- [ ] Run `go test ./... -v` and confirm tests fail
- [ ] Implement all six functions according to spec.md table
- [ ] Run `go test ./... -v` and confirm tests pass
- [ ] Run `golangci-lint run --fix` and fix lint errors
- [ ] Add edge case tests: pow with zero exponent, division by zero handling, chaining operations

---

## 3. Value Advanced Operations (log, exp, relu)

- [ ] Create method stubs `log()`, `exp()`, `relu()` on `Value` (all return `*Value`)
- [ ] Create test functions `TestValueLog`, `TestValueExp`, `TestValueRelu` (verify forward & gradients)
- [ ] Run `go test ./... -v` and confirm tests fail
- [ ] Implement all three methods according to spec.md
- [ ] Run `go test ./... -v` and confirm tests pass
- [ ] Run `golangci-lint run --fix` and fix lint errors
- [ ] Add edge case tests: log of negative/zero, exp overflow, relu on boundaries (0, negative, positive)

---

## 4. Backward Pass (Topological Sort & Backpropagation)

- [ ] Create `backward()` method stub on `Value` (no return)
- [ ] Create test `TestBackwardSimpleAdd` (add two values, backward, check gradients accumulate correctly)
- [ ] Run `go test ./... -v` and confirm test fails
- [ ] Implement `backward()` with DFS topological sort and chain-rule gradient accumulation
  - Use `map[*Value]bool` for visited set
  - Build post-order traversal
  - Reverse iterate and propagate gradients
- [ ] Run `go test ./... -v` and confirm test passes
- [ ] Run `golangci-lint run --fix` and fix lint errors
- [ ] Add critical tests:
  - `TestBackwardSharedNode` (same value used twice, gradients accumulate)
  - `TestBackwardComputation` (longer computation graph, verify all gradients)
  - `go test --race ./...` (ensure no data race on shared nodes)

---

## 5. Helper Function: linear

- [ ] Create `linear(x []*Value, w [][]*Value) []*Value` stub
- [ ] Create test `TestLinear` (3x2 matrix, 2-element vector, verify output is 3-element vector with correct dot products)
- [ ] Run `go test ./... -v` and confirm test fails
- [ ] Implement `linear` as matrix-vector product, no bias
- [ ] Run `go test ./... -v` and confirm test passes
- [ ] Run `golangci-lint run --fix` and fix lint errors
- [ ] Add edge case tests: 1x1 matrix, all-zeros vector, all-zeros matrix, large matrix

---

## 6. Helper Function: softmax

- [ ] Create `softmax(logits []*Value) []*Value` stub
- [ ] Create test `TestSoftmax` (3 logits, verify output sums to ~1, largest logit has largest probability)
- [ ] Run `go test ./... -v` and confirm test fails
- [ ] Implement `softmax` with numeric stability (extract `maxVal` as plain float, don't create Value node)
- [ ] Run `go test ./... -v` and confirm test passes
- [ ] Run `golangci-lint run --fix` and fix lint errors
- [ ] Add critical tests:
  - `TestSoftmaxNumericalStability` (very large logits, should not overflow/NaN)
  - `TestSoftmaxBackward` (backward through softmax, verify gradient flow correct)

---

## 7. Helper Function: rmsnorm

- [ ] Create `rmsnorm(x []*Value) []*Value` stub
- [ ] Create test `TestRMSNorm` (5-element input, verify output has normalized RMS)
- [ ] Run `go test ./... -v` and confirm test fails
- [ ] Implement `rmsnorm` with epsilon 1e-5, no learnable scale parameter
- [ ] Run `go test ./... -v` and confirm test passes
- [ ] Run `golangci-lint run --fix` and fix lint errors
- [ ] Add edge case tests: all-zeros input, single element, very large/small values

---

## 8. Dataset Loading & Tokenization

- [ ] Create functions: `downloadDataset(url, filename string) error`, `loadDocs(filename string) ([]string, error)`, `buildVocab(docs []string) ([]rune, int, int)`
- [ ] Create test `TestDownloadDataset` (mock HTTP or use real URL, verify file exists)
- [ ] Create test `TestLoadDocs` (create temp file, verify parsing and filtering)
- [ ] Create test `TestBuildVocab` (sample input, verify unique chars, BOS value, vocab size)
- [ ] Run `go test ./... -v` and confirm tests fail
- [ ] Implement all three functions (download, parse lines, build sorted unique chars)
- [ ] Run `go test ./... -v` and confirm tests pass
- [ ] Run `golangci-lint run --fix` and fix lint errors
- [ ] Add edge case tests: empty file, file with duplicates, no internet (download fallback)

### 8.A. Shuffle Documents

- [ ] Create `shuffleDocs(docs []string)` function stub (in-place shuffle using global RNG)
- [ ] Create test `TestShuffleDocs` (verify order changes, all elements preserved)
- [ ] Run `go test ./... -v` and confirm test fails
- [ ] Implement `shuffleDocs` using `rng.Shuffle` from Go stdlib
- [ ] Run `go test ./... -v` and confirm test passes
- [ ] Run `golangci-lint run --fix` and fix lint errors
- [ ] Add test verifying shuffle is deterministic with same seed

### 8.B. Encode Function

- [ ] Create `encode(doc string, uchars []rune) []int` function stub
- [ ] Create test `TestEncode` (encode "abc" with known vocab, verify token IDs)
- [ ] Run `go test ./... -v` and confirm test fails
- [ ] Implement `encode` by finding index of each character in `uchars`
- [ ] Run `go test ./... -v` and confirm test passes
- [ ] Run `golangci-lint run --fix` and fix lint errors
- [ ] Add edge case tests: empty string, characters not in vocab (should panic or error)

### 8.C. Decode Function

- [ ] Create `decode(tokens []int, uchars []rune) string` function stub (skip BOS tokens)
- [ ] Create test `TestDecode` (decode token IDs back to string, verify roundtrip)
- [ ] Run `go test ./... -v` and confirm test fails
- [ ] Implement `decode` by mapping each non-BOS token ID to `uchars[id]`
- [ ] Run `go test ./... -v` and confirm test passes
- [ ] Run `golangci-lint run --fix` and fix lint errors
- [ ] Add edge case tests: empty tokens, BOS-only tokens, mixed valid/BOS tokens

---

## 9. Model Parameter Initialization

### 9.A. Matrix Helper Function

- [ ] Create `matrix(nOut, nIn int, std float64) [][]*Value` helper function stub
- [ ] Create test `TestMatrix` (verify shape, verify values follow Normal(0, std))
- [ ] Run `go test ./... -v` and confirm test fails
- [ ] Implement `matrix` using `rng.NormFloat64() * std` for each element
- [ ] Run `go test ./... -v` and confirm test passes
- [ ] Run `golangci-lint run --fix` and fix lint errors
- [ ] Add statistical test: verify mean ≈ 0, stddev ≈ std over large sample

### 9.B. State Dict and Parameter Flattening

- [ ] Create `initStateDict(vocabSize int) StateDict` and `flattenParams(stateDict StateDict) []*Value`
- [ ] Create test `TestInitStateDict` (verify all keys present, correct shapes for given vocabSize)
- [ ] Create test `TestFlattenParams` (verify correct count of parameters)
- [ ] Run `go test ./... -v` and confirm tests fail
- [ ] Implement `initStateDict(vocabSize int)` to create all matrices with Normal(0, 0.08) initialization; `flattenParams` to build flat list
- [ ] Run `go test ./... -v` and confirm tests pass
- [ ] Run `golangci-lint run --fix` and fix lint errors
- [ ] Add edge case tests: different values of `nLayer`, `nEmbd`, `blockSize`, `vocabSize`; verify total parameter count

---

## 10. GPT Forward Pass (Embedding & Transformer Layers)

- [ ] Create `gpt(tokenID, posID int, keys, values [][][]*Value, stateDict StateDict) []*Value` stub
- [ ] Create test `TestGPTEmbedding` (embed a token, verify output is nEmbd-dimensional)
- [ ] Create test `TestGPTAttention` (simple single-head attention, verify output shape)
- [ ] Create test `TestGPTFull` (full forward pass, verify output logits have shape vocabSize)
- [ ] Run `go test ./... -v` and confirm tests fail
- [ ] Implement full GPT forward pass:
  - Token + position embedding + rmsnorm
  - For each layer: attention block (Q/K/V, per-head attention, output projection, residual)
  - For each layer: MLP block (fc1, relu, fc2, residual)
  - Output projection to logits
- [ ] Run `go test ./... -v` and confirm tests pass
- [ ] Run `golangci-lint run --fix` and fix lint errors
- [ ] Add critical tests:
  - `TestGPTBackward` (backward through GPT, verify all gradients non-zero where expected)
  - `TestGPTKVCacheAppend` (verify KV cache is accumulated correctly across positions)
  - `go test --race ./...` (no data races on shared weight nodes)

---

## 11. Adam Optimizer

- [ ] Create `adamOptimizer` type (stores params, m, v, hyperparams)
- [ ] Create `newAdamOptimizer(params []*Value, lr, beta1, beta2, eps float64) *adamOptimizer`
- [ ] Create `(o *adamOptimizer) step(step int)` (one gradient update with bias correction & LR decay)
- [ ] Create test `TestAdamStep` (single param, verify update direction and magnitude)
- [ ] Run `go test ./... -v` and confirm test fails
- [ ] Implement Adam with:
  - First moment (momentum) buffer `m`
  - Second moment (variance) buffer `v`
  - Bias correction: `m_hat = m / (1 - beta1^(step+1))`, `v_hat = v / (1 - beta2^(step+1))`
  - Linear LR decay: `lr_t = lr * (1 - step / numSteps)`
    - ⚠️ **Go integer-division hazard:** cast explicitly — `lr_t = lr * (1.0 - float64(step)/float64(numSteps))`
  - Update: `param -= lr_t * m_hat / (sqrt(v_hat) + eps)`
  - Gradient reset to 0
- [ ] Run `go test ./... -v` and confirm test passes
- [ ] Run `golangci-lint run --fix` and fix lint errors
- [ ] Add critical tests: verify momentum accumulation, verify variance improves convergence

---

## 12. Training Loop

- [ ] Create `train(numSteps int, docs []string, uchars []rune, BOS, vocabSize int, stateDict StateDict, optimizer *adamOptimizer)` (no return, just side effects)
- [ ] Create test `TestTrainSteps` (run for 5 steps on tiny dataset, verify loss decreases or stabilizes)
- [ ] Run `go test ./... -v` and confirm test fails
- [ ] Implement training loop:
  - Per step: select doc (round-robin), encode using `uchars`, wrap with `BOS` tokens
  - Forward pass: call `gpt()` with `tokenID`, `posID`, KV cache, and `stateDict`
  - Compute cross-entropy loss using softmax probabilities and target tokens
  - Average loss: `loss = (1/n) * sum(losses)`
    - ⚠️ **Go integer-division hazard:** use `(1.0 / float64(n)) * sum(losses)` to avoid integer division
  - Backward pass on loss
  - Call `optimizer.step(step)` to update all params in `stateDict`
  - Log progress with format: "step X / Y | loss Z.ZZZZ"
- [ ] Run `go test ./... -v` and confirm test passes
- [ ] Run `golangci-lint run --fix` and fix lint errors
- [ ] Add critical tests: `TestTrainLossDecreases` (over 100 steps on small data, loss should not increase dramatically)

---

## 13. Weighted Random Sampling for Inference

- [ ] Create `weightedChoice(weights []float64) int` function stub
- [ ] Create test `TestWeightedChoice` (verify higher weights selected more often over many samples)
- [ ] Run `go test ./... -v` and confirm test fails
- [ ] Implement `weightedChoice`:
  - Calculate cumulative sum of weights
  - Generate random float in [0, total)
  - Binary search or linear scan to find selected index
- [ ] Run `go test ./... -v` and confirm test passes
- [ ] Run `golangci-lint run --fix` and fix lint errors
- [ ] Add edge case tests: all-zero weights (should handle gracefully), single weight, uniform weights

---

## 13.5. Inference & Sampling

- [ ] Create `sample(temperature float64, maxLen int, uchars []rune, BOS, vocabSize int, stateDict StateDict) string` (return sampled text)
- [ ] Create test `TestSampleLength` (verify output length ≤ maxLen, stops on BOS)
- [ ] Create test `TestSampleTemperature` (temp=0.01 should be more deterministic than temp=2.0)
- [ ] Run `go test ./... -v` and confirm tests fail
- [ ] Implement inference:
  - Initialize fresh KV cache (empty slices for each layer)
  - Start with `tokenID = BOS`
  - For each position: compute logits via `gpt()`, divide by temperature, softmax, weighted sample using `weightedChoice()`
  - If sampled token is `BOS`, stop
  - Otherwise, append `uchars[tokenID]` to result string
  - Stop at max length or BOS token
- [ ] Run `go test ./... -v` and confirm test passes
- [ ] Run `golangci-lint run --fix` and fix lint errors
- [ ] Add edge case tests: temperature extremes, single-step sample, all samples produce valid text

---

## 14. Main Program & Orchestration

- [ ] Create `main()` function stub in `microgpt.go` (no-op initially)
- [ ] Create integration test `TestMainRuns` (verify program starts, downloads data, trains, samples without panicking)
- [ ] Run `go test ./... -v` and confirm test fails
- [ ] Implement `main()` with detailed initialization:
  - Call `initRNG(randomSeed)` to seed global RNG
  - Call `downloadDataset(dataURL, "input.txt")` (skip if file exists)
  - Call `loadDocs("input.txt")` → `docs`
  - Call `shuffleDocs(docs)` (in-place)
  - Call `buildVocab(docs)` → `uchars, BOS, vocabSize`
  - Print `"num docs: {len(docs)}"`
  - Print `"vocab size: {vocabSize}"`
  - Call `initStateDict(vocabSize)` → `stateDict`
  - Call `flattenParams(stateDict)` → `params`
  - Print `"num params: {len(params)}"`
  - Call `newAdamOptimizer(params, learningRate, beta1, beta2, epsAdam)` → `optimizer`
  - Call `train(numSteps, docs, uchars, BOS, vocabSize, stateDict, optimizer)`
  - Print `"\n--- inference (new, hallucinated names) ---"`
  - Call `sample(temperature, blockSize, uchars, BOS, vocabSize, stateDict)` 20 times, print each with formatted index
- [ ] Run `go test ./... -v` and confirm test passes
- [ ] Run `go run .` and verify output (training progress, samples)
- [ ] Run `golangci-lint run --fix` and fix lint errors
- [ ] Final integration test: verify samples are valid names, training loss decreases

### 14.A. Progress Output Formatting

Note: `train()` already produces output after Section 12 is implemented. This section only adds formatting refinements.

- [ ] Add progress output formatting:
  - Training step: use `\r` (carriage return) for same-line overwrite
  - After training loop: print `\n` to move to next line
  - Format: `"step %4d / %4d | loss %.4f"`
- [ ] Add test verifying progress output appears (capture stdout, verify format)

### 14.B. Function Signature Verification

Note: `spec.md` must be updated (see Fix 13 below) before this verification step can succeed. The `gpt()` signature includes the `stateDict` parameter in `todo.md` but not yet in the original `spec.md`.

- [ ] Verify all function signatures match spec.md requirements:
  - `train(numSteps int, docs []string, uchars []rune, BOS, vocabSize int, stateDict StateDict, optimizer *adamOptimizer)`
  - `sample(temperature float64, maxLen int, uchars []rune, BOS, vocabSize int, stateDict StateDict) string`
  - `gpt(tokenID, posID int, keys, values [][][]*Value, stateDict StateDict) []*Value`
- [ ] Verify all tests pass parameters explicitly (no global state dependencies)
- [ ] Run `golangci-lint run --fix` and verify no warnings about unused parameters

---

## Post-Implementation Verification

### Core Verification

- [ ] Run `go test ./... -v` — all tests pass
- [ ] Run `go test --race ./...` — no race conditions
- [ ] Run `go build -o microgpt .` — binary builds successfully
- [ ] Run `go run .` — program executes, trains, samples
- [ ] Verify output matches Python reference (loss values, sample quality, vocabulary)

### Extended Verification

- [ ] Compare training loss curve with Python reference (should be similar trajectory)
- [ ] Compare sample quality with Python reference (should produce name-like strings)
- [ ] Compare vocabulary and BOS values with Python reference (should be identical for same dataset)
- [ ] Verify parameter count: 4192 total parameters for default config
- [ ] Run `go test -cover ./...` and verify coverage is reasonable (target: >80% for core logic)
- [ ] Run `go vet ./...` to check for common Go mistakes
- [ ] Verify no `TODO` or `FIXME` comments remain in code
- [ ] Verify all public functions and types have documentation comments
- [ ] Final code review: ensure all variables/functions follow Go naming conventions (camelCase, exported vs unexported)
