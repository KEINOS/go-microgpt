# go-microgpt

[go-microgpt](https://github.com/KEINOS/go-microgpt/blob/main/microgpt.go) is a Go port of [Andrej Karpathy](https://karpathy.ai/)'s [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) — a minimal GPT implementation to learn transformer internals.

Pure Go, no external dependencies, single-file implementation.

Built for learning—a faithful 1:1 port to understand GPT internals. As the original implementation says, this project is not optimized for efficiency.

**What this project covers:**

- Automatic differentiation (backpropagation through a computation graph)
- Multi-head attention and transformer blocks
- Adam optimizer with learning rate scheduling
- Training and inference loops for sequence models

## Original Implementation

- Python: [gist.github.com/.../microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) [[Rev. 14fb038](https://gist.githubusercontent.com/karpathy/8627fe009c40f57531cb18360106ce95/raw/14fb038816c7aae0bb9342c2dbf1a51dd134a5ff/microgpt.py)]
- Blog: [karpathy.github.io/2026/02/12/microgpt/](https://karpathy.github.io/2026/02/12/microgpt/)

## Quick Start

- Requirements: Go 1.22+

- Run directly:

  ```shellsession
  % # Local run
  % go run ./microgpt
  ```

  ```shellsession
  % # Docker run
  % docker run --rm -v "$(pwd)":/test -w /test golang:1.22-alpine go run ./microgpt.go
  ```

- Build and run:

  ```shellsession
  % go build -o microgpt ./microgpt
  % ./microgpt
  ```

- Run tests:

  ```shellsession
  % # Local run
  % go test ./microgpt -v -race
  ```

  ```shellsession
  % # Docker run
  % docker run --rm -v "$(pwd)":/test -w /test golang:1.22-alpine go test -v ./...
  ```

## Configure

Edit constants in `microgpt/microgpt.go`:

```go
const (
    nLayer    = 1       // transformer layers (depth)
    nEmbd     = 16      // embedding size (width)
    blockSize = 16      // max sequence length per forward pass
    nHead     = 4       // attention heads (must divide nEmbd)
    numSteps  = 1000    // training iterations
    learningRate = 0.01 // Adam learning rate (0.01 recommended)
)
```

- Default: ~3,400 parameters.

**How each affects training:**

| Parameter | Increase | Effect |
| :-------- | :------- | :----- |
| `nLayer` | More layers | Larger model, slower training |
| `nEmbd` | Bigger size | More expressive, higher memory |
| `nHead` | More heads | Better attention patterns, slower |
| `blockSize` | Longer context | Model sees more history |
| `numSteps` | More iterations | Lower loss, longer training |
| `learningRate` | Higher value | Faster convergence, risks instability |

See [Karpathy's blog](https://karpathy.github.io/2026/02/12/microgpt/) for detailed explanations.

## Dataset

Character-level names dataset from [makemore](https://github.com/karpathy/makemore). Auto-downloaded on first run.

## Components

**Included:**

- Autograd system with manual backpropagation
- Multi-head attention, RMSNorm, feed-forward blocks
- Adam optimizer with bias correction
- Autoregressive sampling with temperature scaling
- Character-level tokenization

**Not included (by design):**

- Batching
- Dropout/regularization
- Bias vectors
- Causal masking

## Speed

This section is for reference only.

Even though this Go port runs ~9× faster than Python (due to compiled vs interpreted execution), we focus on faithfully reproducing the original code for learning, not optimizing performance.

```shellsession
% hyperfine "python3 ./ref/microgpt.py" "./microgpt"
Benchmark 1: python3 ./ref/microgpt.py
  Time (mean ± σ):     54.546 s ±  0.931 s

Benchmark 2: ./microgpt
  Time (mean ± σ):      5.928 s ±  0.165 s

Summary: ./microgpt runs 9.20× faster
```

## References

- [たった200行のPythonコードでGPTの学習と推論を動かす【microgpt by A. Karpathy】](https://youtu.be/bR1SyyI7z1k?si=G5XPnE7j-luK53Tu) | [数理の弾丸⚡️京大博士のAI解説](https://www.youtube.com/@mathbullet) @ Youtube (in Japanese)

## License

- [MIT License](LICENSE)
- Authors:
  - [Andrej Karpathy](https://karpathy.ai/) (original Python implementation)
  - [KEINOS](https://github.com/KEINOS) and [the contributors](https://github.com/KEINOS/go-microgpt/graphs/contributors) (Go port)
