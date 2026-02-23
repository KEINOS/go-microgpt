# go-microgpt

A minimal, educational implementation of GPT (Generative Pre-trained Transformer) in Go, ported from [Andrej Karpathy's Python implementation](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95).

The goal is to understand the core transformer algorithm with maximum clarity—all in a single file, dependency-free, with no optimizations beyond what's necessary for comprehension.

## Project Structure

```shellscript
.
├── microgpt/            // Main package containing the implementation
│   ├── microgpt.go      // Complete implementation (~800 lines)
│   └── microgpt_test.go // Comprehensive test suite (39 tests, 82.6% coverage)
├── README.md            // This file
├── go.mod               // Go module file (no external dependencies)
├── spec.md              // Detailed specification matching the Python reference
└── todo.md              // TDD task checklist with implementation steps
```

## Features

### Autograd System

- Manual computation graph tracking
- Automatic differentiation via backward pass
- Support for all core operations: add, mul, pow, log, exp, relu

### Model Architecture

- Multi-head self-attention with scaled dot-product
- Transformer encoder stack (configurable depth)
- Token + position embeddings
- Feed-forward MLP blocks with ReLU
- RMSNorm layer normalization

### Training

- Adam optimizer with momentum and variance correction
- Learning rate decay (linear schedule)
- Cross-entropy loss computation
- Character-level tokenization from dataset

### Inference

- Temperature-controlled sampling
- Autoregressive generation with KV cache
- Stop on special token or max length

## Building

```bash
go build -o microgpt .
```

## Running

```bash
# Run training and inference (downloads dataset on first run)
./microgpt

# Run tests
go test ./... -v

# Run with race detector
go test --race ./...

# Format code
go fmt ./...

# Vet code for common mistakes
go vet ./...
```

## Configuration

Edit constants in `microgpt.go` to adjust:

```go
const (
    nLayer    = 1      // Transformer layers
    nEmbd     = 16     // Embedding dimension
    blockSize = 16     // Context window
    nHead     = 4      // Attention heads
    numSteps  = 1000   // Training iterations
    learningRate = 0.01
    // ... more parameters
)
```

Default configuration: ~3400 parameters

## Expected Output

### Training Output

```bash
num docs: 10000
vocab size: 27
num params: 3424
step    1 / 1000 | loss 3.2956
step    2 / 1000 | loss 3.1845
...
step 1000 / 1000 | loss 1.2345

--- inference (new, hallucinated names) ---
sample  1: emma
sample  2: james
sample  3: oliver
...
```

Loss should decrease over time as the model learns to predict character distributions.

Sample quality improves with more training steps and larger models.

### Test Results

```bash
39/39 tests passing
Coverage: 82.6%
Race conditions: 0 detected
```

## Comparison to Python Reference

The implementation maintains algorithmic equivalence with the Python reference while being idiomatic Go:

| Aspect    | Python   | Go                  |
| --------- | -------- | ------------------- |
| **LOC**   | 201      | 793                 |
| **Tests** | None     | 39 (745 LOC)        |
| **Deps**  | stdlib   | stdlib              |
| **Algo**  | ✅       | ✅                  |
| **Grads** | Autodiff | Manual chain rule   |

## Key Implementation Details

### Autograd (Value Type)

Every scalar is a `*Value` node in the computation graph:

```go
type Value struct {
    Data       float64  // Forward pass result
    Grad       float64  // Accumulated gradient
    children   []*Value // Operands
    localGrads []float64 // Partial derivatives
}
```

Pointer semantics are critical: shared weight nodes must accumulate gradients correctly via `+=` during backward pass.

### Critical Go Pitfalls (Avoided)

1. **Integer Division**: Always cast to float64

   ```go
   // ❌ Wrong: loss = (1/n) * sum(losses)
   // ✅ Right: loss = (1.0/float64(n)) * sum(losses)
   ```

2. **Softmax Numerical Stability**: Extract max as plain float

   ```go
   maxVal := logits[0].Data  // float64, not *Value
   ```

3. **Race Conditions**: Backward pass is sequential (no goroutines)

   ```bash
   go test --race ./...  # Always passes
   ```

## Educational Value

This implementation demonstrates:

- **Transformer Architecture**: Multi-head attention, residual connections, feed-forward blocks
- **Automatic Differentiation**: Backpropagation through arbitrary computation graphs
- **Optimization**: Adam algorithm with bias correction
- **Go Best Practices**: Error handling, idiomatic patterns, memory safety

## Testing

The test suite covers:

- ✅ All arithmetic operations (add, mul, pow, log, exp, relu)
- ✅ Gradient accumulation on shared nodes
- ✅ Topological sort correctness
- ✅ Numerical stability (softmax with large logits)
- ✅ Matrix operations and linear transformations
- ✅ Tokenization and vocabulary building
- ✅ Model initialization and parameter counting
- ✅ Full forward/backward passes through GPT
- ✅ Adam optimizer momentum and learning rate decay
- ✅ Training loop execution
- ✅ Temperature-controlled sampling

Run with: `go test ./... -v`

## Performance

Not optimized (intentional). On a typical laptop:

- Forward pass (1 token): ~0.1ms
- Backward pass: ~0.2ms
- Full training (1000 steps): ~5-10 seconds

For performance benchmarking, run:

```bash
go test ./... -bench=. -benchmem
```

## Debugging

To understand what's happening:

1. **Check loss trajectory**: Should generally decrease during training
2. **Examine gradients**: All parameters should have non-zero gradients
3. **Verify shapes**: Use test cases as reference for tensor dimensions
4. **Compare with Python**: Run same seed in both, compare outputs

## Future Enhancements

Educational additions (not production features):

- Visualization of training loss curve
- Comparison script for Go vs Python outputs
- Profiling to identify bottlenecks
- Beam search for sampling

## License

Educational implementation. Reference original repo: [karpathy/makemore](https://github.com/karpathy/makemore)

## Notes

This is an educational project focused on **clarity over performance**. It prioritizes understanding the algorithm over production efficiency. No external dependencies means you can read and understand every line of math without library abstractions.

The best way to learn: modify the code! Try changing architecture parameters, experiment with different optimizers, or add new features.
