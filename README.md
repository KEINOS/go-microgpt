# go-microgpt

A Go implementation of [Andrej Karpathy](https://karpathy.ai/)'s [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) ─ the simplest way to understand how GPT models train and generate text.

Built in pure Go with no external dependencies in a single file.

Created to deepen my understanding of how transformers/GPT models work at the fundamental level:

- How transformers process information through multiple layers
- How backpropagation computes gradients through the entire network
- How Adam optimizer improves training over standard gradient descent
- How to build neural networks with just basic Go data structures

## Original work by Andrej Karpathy

- [microgpt Python implementation](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) @ gist.github.com
- [Blog post explaining microgpt](https://karpathy.github.io/2026/02/12/microgpt/) @ karpathy.github.io

## Quick Start

```shellsession
% # Simply run
% go run microgpt.go
```

```shellsession
% # Build and run
% go build -o microgpt .
% ./microgpt
```

Run tests:

```shellsession
% go test -v -race .
```

## How to Customize

Edit the constants in `microgpt.go` to adjust the model size and training behavior:

```go
const (
    nLayer    = 1       // number of transformer layers
    nEmbd     = 16      // embedding dimension
    blockSize = 16      // context window size
    nHead     = 4       // number of attention heads
    numSteps  = 1000    // training iterations
    learningRate = 0.01 // learning rate for Adam optimizer
)
```

Default configuration: approximately 3,400 parameters.

## Data

This project uses the names dataset from [Karpathy's makemore project](https://github.com/karpathy/makemore). The dataset is automatically downloaded when you run the program for the first time.

## What's Included

- **Computation Graph Engine** — Manual implementation of automatic differentiation (backpropagation)
- **Transformer Components** — Multi-head self-attention, RMSNorm, feed-forward layers
- **Training** — Adam optimizer with bias correction
- **Inference** — Autoregressive text generation with temperature control
- **Input Format** — Character-level tokenization (trains on individual characters)

**What's not included (by design):**

- Batching
- Dropout or regularization
- Bias terms in linear layers
- Causal masking (trains on single sequences)

## Performance

These numbers are for reference only. This project aims for a 1:1 port of the original, so it is not optimized and is not designed for speed.

```shellsession
% hyperfine "python3 ./ref/microgpt.py" "./microgpt"
Benchmark 1: python3 ./ref/microgpt.py
  Time (mean ± σ):     54.546 s ±  0.931 s    [User: 53.994 s, System: 0.484 s]
  Range (min … max):   53.003 s … 56.012 s    10 runs

Benchmark 2: ./microgpt
  Time (mean ± σ):      5.928 s ±  0.165 s    [User: 12.324 s, System: 0.786 s]
  Range (min … max):    5.796 s …  6.375 s    10 runs

Summary
  ./microgpt ran
    9.20 ± 0.30 times faster than python3 ./ref/microgpt.py
```

## License

- [MIT License](LICENSE)
