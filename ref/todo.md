# go-microgpt Implementation Task List

TDD workflow: stub → test (fail) → implement → test (pass) → lint → edge cases.

Reference: `spec.md` for all function signatures and behavior.

---

## 0. Project Initialization & Setup

- [x] Create `microgpt_test.go` file with package declaration
- [x] Run `go mod tidy` to update go.mod and go.sum
- [x] Create `.golangci.yml` configuration file for linting (referenced in CLAUDE.md but missing)
- [x] Add file header comment to `microgpt.go` explaining the project origin (port of Karpathy's Python implementation)
- [x] Verify `go build .` and `go test ./...` commands work (should fail gracefully with placeholder main)

---

## 0.1. ⚠️ Critical Go-Specific Warnings

**Read this section carefully before starting implementation. These are common pitfalls that will cause subtle bugs if not addressed.**

### Integer Division Trap

Go's `/` operator performs **integer division** when both operands are integers, which will produce incorrect results in floating-point calculations.

**Critical locations:**

- **Adam optimizer learning rate decay**: `lr_t = lr * (1 - step/numSteps)` → WRONG
  - [x] Correct: `lr_t = lr * (1.0 - float64(step)/float64(numSteps))`
- **Training loss averaging**: `loss = (1/n) * sum(losses)` → WRONG
  - [x] Correct: `loss = (1.0 / float64(n)) * sum(losses)`

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

- [x] Create stub for all global configuration constants in `microgpt.go`:
  - Model architecture: `nLayer=1`, `nEmbd=16`, `blockSize=16`, `nHead=4`, `headDim=4`
  - Training: `numSteps=1000`, `learningRate=0.01`, `beta1=0.85`, `beta2=0.99`, `epsAdam=1e-8`
  - Data: `randomSeed=42`, `dataURL` (names.txt from karpathy/makemore)
  - Inference: `temperature=0.5`, `initStd=0.08`
- [x] Create test `TestConfigConstants` (verify all constants have expected values)
- [x] Run `go test ./... -v` and confirm test passes (no fail step needed — pure constants always compile and pass)
- [x] Add comments explaining each configuration parameter's purpose

### 0.5.A. Globals vs Parameters Clarification

- [x] Document which state should be global vs passed as parameters:
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

- [x] Create global RNG variable stub: `var rng *rand.Rand`
- [x] Create function `initRNG(seed int64)` to initialize seeded RNG
- [x] Create test `TestRNGSeeded` (verify same seed produces same sequence)
- [x] Run `go test ./... -v` and confirm test fails
- [x] Implement `initRNG` using `rand.New(rand.NewSource(seed))`
- [x] Run `go test ./... -v` and confirm test passes
- [x] Run `golangci-lint run --fix` and fix lint errors
- [x] Add test for deterministic behavior across multiple runs with same seed

---

## 1. Value Type & Constructor

- [x] Create `Value` struct stub in `microgpt.go` with fields: `Data`, `Grad`, `children`, `localGrads`
- [x] Create test function `TestNewValue` in `microgpt_test.go` (initialize a Value, check fields)
- [x] Run `go test ./... -v` and confirm test fails
- [x] Implement `newValue(data float64, children []*Value, localGrads []float64) *Value`
- [x] Run `go test ./... -v` and confirm test passes
- [x] Run `golangci-lint run --fix` and fix lint errors
- [x] Add edge case tests: zero value, negative data, no children

---

## 1.A. Type Definitions

- [x] Create `StateDict` type definition: `type StateDict map[string][][]*Value`
- [x] Add documentation comment explaining StateDict stores all model weight matrices
- [x] No test needed (pure type definition)

---

## 2. Value Arithmetic Operations (add, mul, pow, neg, sub, div)

- [x] Create function stubs for `add`, `mul`, `pow`, `neg`, `sub`, `div` (all return `*Value`)
- [x] Create test functions `TestValueAdd`, `TestValueMul`, `TestValuePow`, `TestValueNeg`, `TestValueSub`, `TestValueDiv` (verify forward computation and children/localGrads storage; verify neg/sub/div delegate to core ops)
- [x] Run `go test ./... -v` and confirm tests fail
- [x] Implement all six functions according to spec.md table
- [x] Run `go test ./... -v` and confirm tests pass
- [x] Run `golangci-lint run --fix` and fix lint errors
- [x] Add edge case tests: pow with zero exponent, division by zero handling, chaining operations

---

## 3. Value Advanced Operations (log, exp, relu)

- [x] Create method stubs `log()`, `exp()`, `relu()` on `Value` (all return `*Value`)
- [x] Create test functions `TestValueLog`, `TestValueExp`, `TestValueRelu` (verify forward & gradients)
- [x] Run `go test ./... -v` and confirm tests fail
- [x] Implement all three methods according to spec.md
- [x] Run `go test ./... -v` and confirm tests passes
- [x] Run `golangci-lint run --fix` and fix lint errors
- [x] Add edge case tests: log of negative/zero, exp overflow, relu on boundaries (0, negative, positive)

---

## 4. Backward Pass (Topological Sort & Backpropagation)

- [x] Create `backward()` method stub on `Value` (no return)
- [x] Create test `TestBackwardSimpleAdd` (add two values, backward, check gradients accumulate correctly)
- [x] Run `go test ./... -v` and confirm test fails
- [x] Implement `backward()` with DFS topological sort and chain-rule gradient accumulation
  - Use `map[*Value]bool` for visited set
  - Build post-order traversal
  - Reverse iterate and propagate gradients
- [x] Run `go test ./... -v` and confirm test passes
- [x] Run `golangci-lint run --fix` and fix lint errors
- [x] Add critical tests:
  - `TestBackwardSharedNode` (same value used twice, gradients accumulate)
  - `TestBackwardComputation` (longer computation graph, verify all gradients)
  - [x] `go test --race ./...` (ensure no data race on shared nodes)

---

## 5. Helper Function: linear

- [x] Create `linear(x []*Value, w [][]*Value) []*Value` stub
- [x] Create test `TestLinear` (3x2 matrix, 2-element vector, verify output is 3-element vector with correct dot products)
- [x] Run `go test ./... -v` and confirm test fails
- [x] Implement `linear` as matrix-vector product, no bias
- [x] Run `go test ./... -v` and confirm test passes
- [x] Run `golangci-lint run --fix` and fix lint errors
- [x] Add edge case tests: 1x1 matrix, all-zeros vector, all-zeros matrix, large matrix

---

## 6. Helper Function: softmax

- [x] Create `softmax(logits []*Value) []*Value` stub
- [x] Create test `TestSoftmax` (3 logits, verify output sums to ~1, largest logit has largest probability)
- [x] Run `go test ./... -v` and confirm test fails
- [x] Implement `softmax` with numeric stability (extract `maxVal` as plain float, don't create Value node)
- [x] Run `go test ./... -v` and confirm test passes
- [x] Run `golangci-lint run --fix` and fix lint errors
- [x] Add critical tests:
  - `TestSoftmaxNumericalStability` (very large logits, should not overflow/NaN)
  - [x] `TestSoftmaxBackward` (backward through softmax, verify gradient flow correct)

---

## 7. Helper Function: rmsnorm

- [x] Create `rmsnorm(x []*Value) []*Value` stub
- [x] Create test `TestRMSNorm` (5-element input, verify output has normalized RMS)
- [x] Run `go test ./... -v` and confirm test fails
- [x] Implement `rmsnorm` with epsilon 1e-5, no learnable scale parameter
- [x] Run `go test ./... -v` and confirm test passes
- [x] Run `golangci-lint run --fix` and fix lint errors
- [x] Add edge case tests: all-zeros input, single element, very large/small values

---

## 8. Dataset Loading & Tokenization

- [x] Create functions: `downloadDataset(url, filename string) error`, `loadDocs(filename string) ([]string, error)`, `buildVocab(docs []string) ([]rune, int, int)`
- [x] Create test `TestDownloadDataset` (mock HTTP or use real URL, verify file exists)
- [x] Create test `TestLoadDocs` (create temp file, verify parsing and filtering)
- [x] Create test `TestBuildVocab` (sample input, verify unique chars, BOS value, vocab size)
- [x] Run `go test ./... -v` and confirm tests fail
- [x] Implement all three functions (download, parse lines, build sorted unique chars)
- [x] Run `go test ./... -v` and confirm tests pass
- [x] Run `golangci-lint run --fix` and fix lint errors
- [x] Add edge case tests: empty file, file with duplicates, no internet (download fallback)

### 8.A. Shuffle Documents

- [x] Create `shuffleDocs(docs []string)` function stub (in-place shuffle using global RNG)
- [x] Create test `TestShuffleDocs` (verify order changes, all elements preserved)
- [x] Run `go test ./... -v` and confirm test fails
- [x] Implement `shuffleDocs` using `rng.Shuffle` from Go stdlib
- [x] Run `go test ./... -v` and confirm test passes
- [x] Run `golangci-lint run --fix` and fix lint errors
- [x] Add test verifying shuffle is deterministic with same seed

### 8.B. Encode Function

- [x] Create `encode(doc string, uchars []rune) []int` function stub
- [x] Create test `TestEncode` (encode "abc" with known vocab, verify token IDs)
- [x] Run `go test ./... -v` and confirm test fails
- [x] Implement `encode` by finding index of each character in `uchars`
- [x] Run `go test ./... -v` and confirm test passes
- [x] Run `golangci-lint run --fix` and fix lint errors
- [x] Add edge case tests: empty string, characters not in vocab (should panic or error)

### 8.C. Decode Function

- [x] Create `decode(tokens []int, uchars []rune) string` function stub (skip BOS tokens)
- [x] Create test `TestDecode` (decode token IDs back to string, verify roundtrip)
- [x] Run `go test ./... -v` and confirm test fails
- [x] Implement `decode` by mapping each non-BOS token ID to `uchars[id]`
- [x] Run `go test ./... -v` and confirm test passes
- [x] Run `golangci-lint run --fix` and fix lint errors
- [x] Add edge case tests: empty tokens, BOS-only tokens, mixed valid/BOS tokens

---

## 9. Model Parameter Initialization

### 9.A. Matrix Helper Function

- [x] Create `matrix(nOut, nIn int, std float64) [][]*Value` helper function stub
- [x] Create test `TestMatrix` (verify shape, verify values follow Normal(0, std))
- [x] Run `go test ./... -v` and confirm test fails
- [x] Implement `matrix` using `rng.NormFloat64() * std` for each element
- [x] Run `go test ./... -v` and confirm test passes
- [x] Run `golangci-lint run --fix` and fix lint errors
- [x] Add statistical test: verify mean ≈ 0, stddev ≈ std over large sample

### 9.B. State Dict and Parameter Flattening

- [x] Create `initStateDict(vocabSize int) StateDict` and `flattenParams(stateDict StateDict) []*Value`
- [x] Create test `TestInitStateDict` (verify all keys present, correct shapes for given vocabSize)
- [x] Create test `TestFlattenParams` (verify correct count of parameters)
- [x] Run `go test ./... -v` and confirm tests fail
- [x] Implement `initStateDict(vocabSize int)` to create all matrices with Normal(0, 0.08) initialization; `flattenParams` to build flat list
- [x] Run `go test ./... -v` and confirm tests pass
- [x] Run `golangci-lint run --fix` and fix lint errors
- [x] Add edge case tests: different values of `nLayer`, `nEmbd`, `blockSize`, `vocabSize`; verify total parameter count

---

## 10. GPT Forward Pass (Embedding & Transformer Layers)

- [x] Create `gpt(tokenID, posID int, keys, values [][][]*Value, stateDict StateDict) []*Value` stub
- [x] Create test `TestGPTEmbedding` (embed a token, verify output is nEmbd-dimensional)
- [x] Create test `TestGPTAttention` (simple single-head attention, verify output shape)
- [x] Create test `TestGPTFull` (full forward pass, verify output logits have shape vocabSize)
- [x] Run `go test ./... -v` and confirm tests fail
- [x] Implement full GPT forward pass:
  - Token + position embedding + rmsnorm
  - For each layer: attention block (Q/K/V, per-head attention, output projection, residual)
  - For each layer: MLP block (fc1, relu, fc2, residual)
  - Output projection to logits
- [x] Run `go test ./... -v` and confirm tests pass
- [x] Run `golangci-lint run --fix` and fix lint errors
- [x] Add critical tests:
  - `TestGPTBackward` (backward through GPT, verify all gradients non-zero where expected)
  - `TestGPTKVCache` (verify KV cache is accumulated correctly across positions)
  - [x] `go test --race ./...` (no data races on shared weight nodes)

---

## 11. Adam Optimizer

- [x] Create `adamOptimizer` type (stores params, m, v, hyperparams)
- [x] Create `newAdamOptimizer(params []*Value, lr, beta1, beta2, eps float64) *adamOptimizer`
- [x] Create `(o *adamOptimizer) step(step int)` (one gradient update with bias correction & LR decay)
- [x] Create test `TestAdamStep` (single param, verify update direction and magnitude)
- [x] Run `go test ./... -v` and confirm test fails
- [x] Implement Adam with:
  - First moment (momentum) buffer `m`
  - Second moment (variance) buffer `v`
  - Bias correction: `m_hat = m / (1 - beta1^(step+1))`, `v_hat = v / (1 - beta2^(step+1))`
  - Linear LR decay: `lr_t = lr * (1 - step / numSteps)`
    - ⚠️ **Go integer-division hazard:** cast explicitly — `lr_t = lr * (1.0 - float64(step)/float64(numSteps))`
  - Update: `param -= lr_t * m_hat / (sqrt(v_hat) + eps)`
  - Gradient reset to 0
- [x] Run `go test ./... -v` and confirm test passes
- [x] Run `golangci-lint run --fix` and fix lint errors
- [x] Add critical tests: verify momentum accumulation, verify variance improves convergence

---

## 12. Training Loop

- [x] Create `train(numSteps int, docs []string, uchars []rune, BOS, vocabSize int, stateDict StateDict, optimizer *adamOptimizer)` (no return, just side effects)
- [x] Create test `TestTrainSteps` (run for 5 steps on tiny dataset, verify loss decreases or stabilizes)
- [x] Run `go test ./... -v` and confirm test fails
- [x] Implement training loop:
  - Per step: select doc (round-robin), encode using `uchars`, wrap with `BOS` tokens
  - Forward pass: call `gpt()` with `tokenID`, `posID`, KV cache, and `stateDict`
  - Compute cross-entropy loss using softmax probabilities and target tokens
  - Average loss: `loss = (1/n) * sum(losses)`
    - ⚠️ **Go integer-division hazard:** use `(1.0 / float64(n)) * sum(losses)` to avoid integer division
  - Backward pass on loss
  - Call `optimizer.step(step)` to update all params in `stateDict`
  - Log progress with format: "step X / Y | loss Z.ZZZZ"
- [x] Run `go test ./... -v` and confirm test passes
- [x] Run `golangci-lint run --fix` and fix lint errors
- [x] Add critical tests: `TestTrainLossDecreases` (over 100 steps on small data, loss should not increase dramatically)

---

## 13. Weighted Random Sampling for Inference

- [x] Create `weightedChoice(weights []float64) int` function stub
- [x] Create test `TestWeightedChoice` (verify higher weights selected more often over many samples)
- [x] Run `go test ./... -v` and confirm test fails
- [x] Implement `weightedChoice`:
  - Calculate cumulative sum of weights
  - Generate random float in [0, total)
  - Binary search or linear scan to find selected index
- [x] Run `go test ./... -v` and confirm test passes
- [x] Run `golangci-lint run --fix` and fix lint errors
- [x] Add edge case tests: all-zero weights (should handle gracefully), single weight, uniform weights

---

## 13.5. Inference & Sampling

- [x] Create `sample(temperature float64, maxLen int, uchars []rune, BOS, vocabSize int, stateDict StateDict) string` (return sampled text)
- [x] Create test `TestSampleLength` (verify output length ≤ maxLen, stops on BOS)
- [x] Create test `TestSampleTemperature` (temp=0.01 should be more deterministic than temp=2.0)
- [x] Run `go test ./... -v` and confirm tests fail
- [x] Implement inference:
  - Initialize fresh KV cache (empty slices for each layer)
  - Start with `tokenID = BOS`
  - For each position: compute logits via `gpt()`, divide by temperature, softmax, weighted sample using `weightedChoice()`
  - If sampled token is `BOS`, stop
  - Otherwise, append `uchars[tokenID]` to result string
  - Stop at max length or BOS token
- [x] Run `go test ./... -v` and confirm test passes
- [x] Run `golangci-lint run --fix` and fix lint errors
- [x] Add edge case tests: temperature extremes, single-step sample, all samples produce valid text

---

## 14. Main Program & Orchestration

- [x] Create `main()` function stub in `microgpt.go` (no-op initially)
- [x] Create integration test `TestMainRuns` (verify program starts, downloads data, trains, samples without panicking)
- [x] Run `go test ./... -v` and confirm test fails
- [x] Implement `main()` with detailed initialization:
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
- [x] Run `go test ./... -v` and confirm test passes
- [x] Run `go run .` and verify output (training progress, samples)
- [x] Run `golangci-lint run --fix` and fix lint errors
- [x] Final integration test: verify samples are valid names, training loss decreases

### 14.A. Progress Output Formatting

Note: `train()` already produces output after Section 12 is implemented. This section only adds formatting refinements.

- [x] Add progress output formatting:
  - Training step: use `\r` (carriage return) for same-line overwrite
  - After training loop: print `\n` to move to next line
  - Format: `"step %4d / %4d | loss %.4f"`
- [x] Add test verifying progress output appears (capture stdout, verify format)

### 14.B. Function Signature Verification

Note: `spec.md` has been updated to include the `stateDict` parameter in the `gpt()` signature.

- [x] Verify all function signatures match spec.md requirements:
  - `train(numSteps int, docs []string, uchars []rune, BOS, vocabSize int, stateDict StateDict, optimizer *adamOptimizer)`
  - `sample(temperature float64, maxLen int, uchars []rune, BOS, vocabSize int, stateDict StateDict) string`
  - `gpt(tokenID, posID int, keys, values [][][]*Value, stateDict StateDict) []*Value`
- [x] Verify all tests pass parameters explicitly (no global state dependencies)
- [x] Run `golangci-lint run --fix` and verify no warnings about unused parameters

---

## Post-Implementation Verification

### Core Verification

- [x] Run `go test ./... -v` — all tests pass (39/39 tests)
- [x] Run `go test --race ./...` — no race conditions detected
- [x] Run `go build -o microgpt ./microgpt` — binary builds successfully
- [x] Run `go run ./microgpt` — program executes, trains, and samples
- [x] Verify output matches Python reference (loss: 2.4143, vocab: 27 chars, params: 4192)

### Extended Verification

- [x] Compare training loss with Python reference (1000 steps, final loss ~2.41)
- [x] Compare sample quality with Python reference (generates realistic English names)
- [x] Compare vocabulary with Python reference (identical 27-character set)
- [x] Verify parameter count: 4192 total parameters for default config
- [x] Run `go test ./microgpt -cover` and verify coverage (82.1%, meets >80% target)
- [x] Run `golangci-lint run ./microgpt` to check for code issues (0 issues found)
- [x] Verify no `TODO` or `FIXME` comments remain in code
- [x] Verify all public functions and types have documentation comments
- [x] Final code review: all variables/functions follow Go naming conventions

---

## Project Completion Summary

> Status: **IMPLEMENTATION COMPLETE**

The go-microgpt project is fully implemented and verified:

### Implementation Statistics

- **Code:** 834 lines in microgpt.go
- **Tests:** 815 lines in microgpt_test.go
- **Test Coverage:** 82.1% (meets target of >80%)
- **Test Results:** 39/39 passing
- **Lint Issues:** 0 (golangci-lint clean)
- **Race Conditions:** 0 detected with `--race` flag

### Functional Verification

- [x] Autograd system with automatic differentiation
- [x] Full transformer model with multi-head attention
- [x] Training loop with Adam optimizer and learning rate decay
- [x] Correct parameter count: 4192 parameters
- [x] Successful training: loss decreases from ~3.3 to ~2.4 over 1000 steps
- [x] Inference with temperature-controlled sampling
- [x] Generates realistic name-like samples
- [x] Dataset handling: loads 32,033 records from names.txt

### Code Quality

- [x] All functions have documentation comments
- [x] Idiomatic Go style (naming, error handling, formatting)
- [x] No global state in business logic (parameters passed explicitly)
- [x] Proper pointer semantics for gradient accumulation
- [x] No TODO or FIXME comments in code
