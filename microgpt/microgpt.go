// Package main implements microgpt - a minimal, dependency-free educational implementation of a GPT language model.
// This is a port of Andrej Karpathy's Python implementation to Go.
// All code is in a single file to demonstrate the core transformer algorithm with maximum clarity.
//
// Reference: https://github.com/karpathy/makemore
package main

import (
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"os"
	"slices"
	"sort"
	"strings"
)

// ============================================================================
// Configuration & Constants
// ============================================================================

// Magic Numbers.
const (
	zeroDivisionEps   = 1e-5 // Small constant to prevent division by zero
	mlpExpansionRatio = 4    // MLP hidden layer expansion ratio (nEmbd -> 4*nEmbd -> nEmbd)
)

// Model architecture configuration.
const (
	nLayer    = 1             // Number of transformer layers
	nEmbd     = 16            // Embedding dimension
	blockSize = 16            // Maximum context window length
	nHead     = 4             // Number of attention heads
	headDim   = nEmbd / nHead // Dimension per head = 4
)

// Training configuration.
const (
	numSteps     = 1000 // Training steps
	learningRate = 0.01 // Initial learning rate
	beta1        = 0.85 // Adam first-moment decay
	beta2        = 0.99 // Adam second-moment decay
	epsAdam      = 1e-8 // Adam epsilon
)

// Data and inference configuration.
const (
	randomSeed  = 42 // Global RNG seed
	dataURL     = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
	temperature = 0.5  // Sampling temperature for inference
	initStd     = 0.08 // Std dev for weight initialization
)

// Global RNG - seeded once at startup.
var rng *rand.Rand

// ============================================================================
// Value Type - Autograd Node
// ============================================================================

// Value represents a scalar node in a computation graph for automatic differentiation.
// All scalars in computation must be *Value (not Value) to support gradient accumulation
// on shared weight nodes during backward pass.
type Value struct {
	Data       float64   // Forward pass scalar value
	Grad       float64   // Gradient accumulated during backward pass
	children   []*Value  // Child nodes (operands)
	localGrads []float64 // Local derivatives w.r.t. children (chain rule coefficients)
}

// StateDict stores all model weight matrices indexed by name.
type StateDict map[string][][]*Value

// ============================================================================
//  Main Function (Training & Inference)
// ============================================================================

func main() {
	// Initialize global Random Number Generator
	initRNG(randomSeed)

	// Download dataset if needed
	err := downloadDataset(dataURL, "input.txt")
	if err != nil {
		fmt.Fprintf(os.Stderr, "error downloading dataset: %v\n", err)
		os.Exit(1)
	}

	// Load documents
	docs, err := loadDocs("input.txt")
	if err != nil {
		fmt.Fprintf(os.Stderr, "error loading docs: %v\n", err)
		os.Exit(1)
	}

	// Shuffle documents
	shuffleDocs(docs)
	fmt.Printf("num docs: %d\n", len(docs))

	// Build vocabulary
	uchars, BOS, vocabSize := buildVocab(docs)
	fmt.Printf("vocab size: %d\n", vocabSize)

	// Initialize model
	stateDict := initStateDict(vocabSize)
	params := flattenParams(stateDict)
	fmt.Printf("num params: %d\n", len(params))

	// Create optimizer
	optimizer := newAdamOptimizer(params, learningRate, beta1, beta2, epsAdam)

	// Training loop
	train(numSteps, docs, uchars, BOS, vocabSize, stateDict, optimizer)

	// Inference
	fmt.Println("\n--- inference (new, hallucinated names) ---")

	for sampleIdx := range 20 {
		result := sample(temperature, blockSize, uchars, BOS, vocabSize, stateDict)
		fmt.Printf("sample %2d: %s\n", sampleIdx+1, result)
	}
}

// ============================================================================
// RNG Initialization
// ============================================================================

// initRNG initializes the global random number generator with a seed.
func initRNG(seed int64) {
	source := rand.NewSource(seed)
	rng = rand.New(source)
}

// ============================================================================
// Section 1: Value Constructor
// ============================================================================

// newValue creates a new Value node.
func newValue(data float64, children []*Value, localGrads []float64) *Value {
	return &Value{
		Data:       data,
		Grad:       0,
		children:   children,
		localGrads: localGrads,
	}
}

// ============================================================================
// Section 2: Value Arithmetic Operations
// ============================================================================

// add returns a + b.
func add(a, b *Value) *Value {
	return newValue(a.Data+b.Data, []*Value{a, b}, []float64{1, 1})
}

// mul returns a * b.
func mul(a, b *Value) *Value {
	return newValue(a.Data*b.Data, []*Value{a, b}, []float64{b.Data, a.Data})
}

// pow returns a ^ exp.
func pow(a *Value, exp float64) *Value {
	return newValue(
		math.Pow(a.Data, exp),
		[]*Value{a},
		[]float64{exp * math.Pow(a.Data, exp-1)},
	)
}

// neg returns -a.
func neg(a *Value) *Value {
	return mul(a, newValue(-1, nil, nil))
}

// sub returns a - b.
func sub(a, b *Value) *Value {
	return add(a, neg(b))
}

// div returns a / b.
func div(a, b *Value) *Value {
	return mul(a, pow(b, -1))
}

// ============================================================================
// Section 3: Value Advanced Operations
// ============================================================================

// log returns ln(v).
func (v *Value) log() *Value {
	return newValue(
		math.Log(v.Data),
		[]*Value{v},
		[]float64{1 / v.Data},
	)
}

// exp returns e^v.
func (v *Value) exp() *Value {
	return newValue(
		math.Exp(v.Data),
		[]*Value{v},
		[]float64{math.Exp(v.Data)},
	)
}

// relu returns max(0, v).
func (v *Value) relu() *Value {
	relu := 0.0
	if v.Data > 0 {
		relu = 1.0
	}

	return newValue(
		math.Max(0, v.Data),
		[]*Value{v},
		[]float64{relu},
	)
}

// ============================================================================
// Section 4: Backward Pass - Topological Sort & Backpropagation
// ============================================================================

// backward performs reverse-mode automatic differentiation on the computation graph.
func (v *Value) backward() {
	// Build topological order via DFS post-order traversal
	visited := make(map[*Value]bool)
	topo := make([]*Value, 0)

	var buildTopo func(*Value)

	buildTopo = func(node *Value) {
		if visited[node] {
			return
		}

		visited[node] = true
		for _, child := range node.children {
			buildTopo(child)
		}

		topo = append(topo, node)
	}

	buildTopo(v)

	// Set root gradient to 1
	v.Grad = 1.0

	// Propagate gradients in reverse topological order
	for i := len(topo) - 1; i >= 0; i-- {
		node := topo[i]
		for j, child := range node.children {
			child.Grad += node.localGrads[j] * node.Grad
		}
	}
}

// ============================================================================
// Section 5: Helper Function - linear (matrix-vector product)
// ============================================================================

// linear computes y = W @ x (matrix-vector product, no bias)
// w is shape [nOut, nIn], x is shape [nIn], returns shape [nOut].
func linear(x []*Value, w [][]*Value) []*Value {
	out := make([]*Value, len(w))
	for i := range w {
		sum := newValue(0, nil, nil)
		for j := range x {
			sum = add(sum, mul(w[i][j], x[j]))
		}

		out[i] = sum
	}

	return out
}

// ============================================================================
// Section 6: Helper Function - softmax
// ============================================================================

// softmax computes numerically stable softmax over logits.
func softmax(logits []*Value) []*Value {
	// Extract max as plain float64 for numerical stability
	maxVal := logits[0].Data
	for _, v := range logits[1:] {
		if v.Data > maxVal {
			maxVal = v.Data
		}
	}

	// Compute exps and their sum
	exps := make([]*Value, len(logits))
	maxValNode := newValue(maxVal, nil, nil)
	total := newValue(0, nil, nil)

	for i, v := range logits {
		exps[i] = sub(v, maxValNode).exp()
		total = add(total, exps[i])
	}

	// Normalize
	out := make([]*Value, len(exps))
	for i := range exps {
		out[i] = div(exps[i], total)
	}

	return out
}

// ============================================================================
// Section 7: Helper Function - rmsnorm
// ============================================================================

// rmsnorm computes root-mean-square normalization.
// Note: This implementation omits the learnable scale parameter (γ/gamma)
// to match the Python reference (microgpt.py), maintaining 1:1 parity for
// educational clarity. Standard RMSNorm includes γ, but it's omitted here
// for simplicity, following Andrej Karpathy's minimal design.
func rmsnorm(x []*Value) []*Value {
	// ms = sum(xi * xi) / len(x)
	sumSq := newValue(0, nil, nil)
	for _, xi := range x {
		sumSq = add(sumSq, mul(xi, xi))
	}

	ms := mul(sumSq, newValue(1.0/float64(len(x)), nil, nil))

	// scale = pow(ms + 1e-5, -0.5)
	scale := pow(add(ms, newValue(zeroDivisionEps, nil, nil)), -0.5)

	// return [xi * scale for xi in x]
	out := make([]*Value, len(x))
	for i := range x {
		out[i] = mul(x[i], scale)
	}

	return out
}

// ============================================================================
// Section 8: Dataset Loading & Tokenization
// ============================================================================

// downloadDataset downloads the dataset from a URL if it doesn't exist locally.
func downloadDataset(url, filename string) error {
	if _, err := os.Stat(filename); err == nil {
		// File already exists
		return nil
	}

	resp, err := http.Get(url)
	if err != nil {
		return err
	}

	defer func() {
		_ = resp.Body.Close() // Ignore close error in defer
	}()

	file, err := os.Create(filename)
	if err != nil {
		return err
	}

	defer func() {
		_ = file.Close() // Ignore close error in defer
	}()

	_, err = io.Copy(file, resp.Body)

	return err
}

// loadDocs reads all lines from a file, strips whitespace, and returns non-empty lines.
func loadDocs(filename string) ([]string, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	lines := strings.Split(string(data), "\n")

	var docs []string

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line != "" {
			docs = append(docs, line)
		}
	}

	return docs, nil
}

// buildVocab extracts unique characters from docs, sorts them, and returns:
// uchars (sorted unique characters), BOS (special token ID), vocabSize (total tokens).
func buildVocab(docs []string) ([]rune, int, int) {
	// Collect all unique characters
	charSet := make(map[rune]bool)

	for _, doc := range docs {
		for _, ch := range doc {
			charSet[ch] = true
		}
	}

	// Convert to sorted slice
	uchars := make([]rune, 0, len(charSet))
	for ch := range charSet {
		uchars = append(uchars, ch)
	}

	slices.Sort(uchars)

	BOS := len(uchars)
	vocabSize := len(uchars) + 1

	return uchars, BOS, vocabSize
}

// buildVocabIndex builds a rune->index map for fast token lookup.
func buildVocabIndex(uchars []rune) map[rune]int {
	index := make(map[rune]int, len(uchars))
	for i, ch := range uchars {
		index[ch] = i
	}

	return index
}

// shuffleDocs shuffles documents in-place using the global RNG.
func shuffleDocs(docs []string) {
	rng.Shuffle(len(docs), func(i, j int) {
		docs[i], docs[j] = docs[j], docs[i]
	})
}

// encode converts a string to token IDs using a precomputed vocab index.
func encode(doc string, vocabIndex map[rune]int) []int {
	tokens := make([]int, len(doc))
	for i, ch := range doc {
		index, ok := vocabIndex[ch]
		if !ok {
			panic(fmt.Sprintf("character %c not in vocabulary", ch))
		}

		tokens[i] = index
	}

	return tokens
}

// decode converts token IDs back to a string, skipping BOS tokens.
func decode(tokens []int, uchars []rune, BOS int) string {
	var result strings.Builder

	for _, tokenID := range tokens {
		if tokenID != BOS && tokenID < len(uchars) {
			result.WriteRune(uchars[tokenID])
		}
	}

	return result.String()
}

// ============================================================================
// Section 9: Model Parameter Initialization
// ============================================================================

// matrix creates a matrix of shape [nOut, nIn] initialized with Normal(0, std).
func matrix(nOut, nIn int, std float64) [][]*Value {
	mat := make([][]*Value, nOut)
	for i := range mat {
		mat[i] = make([]*Value, nIn)
		for j := range mat[i] {
			// rng.NormFloat64() returns standard normal; multiply by std.
			// Note: Go and Python use different Gaussian generators, so exact weights differ.
			mat[i][j] = newValue(rng.NormFloat64()*std, nil, nil)
		}
	}

	return mat
}

// initStateDict creates and initializes all model weight matrices.
func initStateDict(vocabSize int) StateDict {
	sd := make(StateDict)

	// Embedding tables
	sd["wte"] = matrix(vocabSize, nEmbd, initStd) // token embeddings
	sd["wpe"] = matrix(blockSize, nEmbd, initStd) // position embeddings

	// Note: Modern GPTs use weight tying (lm_head = wte.T) to reduce parameters
	// and improve performance, but this implementation keeps them separate to
	// match the Python reference (microgpt.py) for educational clarity and 1:1 parity.
	sd["lm_head"] = matrix(vocabSize, nEmbd, initStd) // output logits projection

	// Transformer layers
	for i := range nLayer {
		key := fmt.Sprintf("layer%d", i)
		// Attention weights
		sd[key+".attn_wq"] = matrix(nEmbd, nEmbd, initStd) // query projection
		sd[key+".attn_wk"] = matrix(nEmbd, nEmbd, initStd) // key projection
		sd[key+".attn_wv"] = matrix(nEmbd, nEmbd, initStd) // value projection
		sd[key+".attn_wo"] = matrix(nEmbd, nEmbd, initStd) // output projection
		// MLP weights
		sd[key+".mlp_fc1"] = matrix(mlpExpansionRatio*nEmbd, nEmbd, initStd) // expand layer
		sd[key+".mlp_fc2"] = matrix(nEmbd, mlpExpansionRatio*nEmbd, initStd) // contract layer
	}

	return sd
}

// flattenParams extracts all parameters from stateDict into a single flat list.
func flattenParams(stateDict StateDict) []*Value {
	var params []*Value

	keys := make([]string, 0, len(stateDict))
	for key := range stateDict {
		keys = append(keys, key)
	}

	// Use deterministic key order to keep optimizer buffers stable across runs.
	sort.Strings(keys)

	for _, key := range keys {
		mat := stateDict[key]
		for _, row := range mat {
			params = append(params, row...)
		}
	}

	return params
}

// ============================================================================
// Section 10: GPT Forward Pass
// ============================================================================

// gpt computes the forward pass of the GPT model
// Returns logits of shape [vocabSize].
func gpt(tokenID, posID int, keys, values [][][]*Value, stateDict StateDict) []*Value {
	// Step 1: Token + Position Embedding
	tokEmb := stateDict["wte"][tokenID] // [nEmbd]
	posEmb := stateDict["wpe"][posID]   // [nEmbd]

	// Element-wise add embeddings
	x := make([]*Value, nEmbd)
	for i := range nEmbd {
		x[i] = add(tokEmb[i], posEmb[i])
	}

	// Apply RMSNorm
	x = rmsnorm(x)

	// Step 2: Transformer Layers
	for li := range nLayer {
		layerKey := fmt.Sprintf("layer%d", li)
		// ============================================================
		// Attention Sub-block
		// ============================================================
		xRes := x
		x = rmsnorm(x)

		// Compute Q, K, V
		q := linear(x, stateDict[layerKey+".attn_wq"]) // [nEmbd]
		k := linear(x, stateDict[layerKey+".attn_wk"]) // [nEmbd]
		v := linear(x, stateDict[layerKey+".attn_wv"]) // [nEmbd]

		// Append to KV cache
		keys[li] = append(keys[li], k)
		values[li] = append(values[li], v)

		// Per-head attention
		xAttn := make([]*Value, 0, nEmbd)

		for h := range nHead {
			hs := h * headDim
			qH := q[hs : hs+headDim]
			kH := make([][]*Value, len(keys[li]))

			vH := make([][]*Value, len(values[li]))
			// We only have past tokens in KV cache, so no explicit causal mask is needed.
			// If you later batch full sequences, add a mask to block future tokens.
			for t := range len(keys[li]) {
				kH[t] = keys[li][t][hs : hs+headDim]
				vH[t] = values[li][t][hs : hs+headDim]
			}

			// Compute scaled dot-product attention scores
			scores := make([]*Value, len(keys[li]))
			for t := range len(keys[li]) {
				sum := newValue(0, nil, nil)
				for j := range headDim {
					sum = add(sum, mul(qH[j], kH[t][j]))
				}

				scores[t] = mul(sum, newValue(1.0/math.Sqrt(float64(headDim)), nil, nil))
			}

			// Softmax to get attention weights
			attnW := softmax(scores)

			// Weighted value aggregation
			headOut := make([]*Value, headDim)
			for j := range headDim {
				headOut[j] = newValue(0, nil, nil)
				for t := range len(values[li]) {
					headOut[j] = add(headOut[j], mul(attnW[t], vH[t][j]))
				}
			}

			xAttn = append(xAttn, headOut...)
		}

		// Output projection
		x = linear(xAttn, stateDict[layerKey+".attn_wo"])

		// Residual connection
		for i := range nEmbd {
			x[i] = add(x[i], xRes[i])
		}

		// ============================================================
		// MLP Sub-block
		// ============================================================
		xRes = x
		x = rmsnorm(x)

		// Expand layer
		x = linear(x, stateDict[layerKey+".mlp_fc1"]) // [4*nEmbd]

		// ReLU activation
		for i := range x {
			x[i] = x[i].relu()
		}

		// Contract layer
		x = linear(x, stateDict[layerKey+".mlp_fc2"]) // [nEmbd]

		// Residual connection
		for i := range nEmbd {
			x[i] = add(x[i], xRes[i])
		}
	}

	// Step 3: Output logits
	logits := linear(x, stateDict["lm_head"])

	return logits
}

// ============================================================================
// Section 11: Adam Optimizer
// ============================================================================

// adamOptimizer implements the Adam optimization algorithm.
type adamOptimizer struct {
	params []*Value
	m      []float64 // First moment (momentum)
	v      []float64 // Second moment (variance)
	lr     float64   // Learning rate
	beta1  float64   // First moment decay
	beta2  float64   // Second moment decay
	eps    float64   // Epsilon for numerical stability
}

// newAdamOptimizer creates a new Adam optimizer.
func newAdamOptimizer(params []*Value, lr, beta1, beta2, eps float64) *adamOptimizer {
	return &adamOptimizer{
		params: params,
		m:      make([]float64, len(params)),
		v:      make([]float64, len(params)),
		lr:     lr,
		beta1:  beta1,
		beta2:  beta2,
		eps:    eps,
	}
}

// step performs one optimization step with learning rate decay.
func (o *adamOptimizer) step(stepNum int) {
	// Learning rate decay: lr_t = lr * (1 - step / numSteps)
	// CRITICAL: Avoid integer division - cast to float64
	lrT := o.lr * (1.0 - float64(stepNum)/float64(numSteps))

	for i, p := range o.params {
		// Update biased first moment
		o.m[i] = o.beta1*o.m[i] + (1-o.beta1)*p.Grad

		// Update biased second moment
		o.v[i] = o.beta2*o.v[i] + (1-o.beta2)*p.Grad*p.Grad

		// Compute bias-corrected first moment
		mHat := o.m[i] / (1 - math.Pow(o.beta1, float64(stepNum+1)))

		// Compute bias-corrected second moment
		vHat := o.v[i] / (1 - math.Pow(o.beta2, float64(stepNum+1)))

		// Update parameter
		p.Data -= lrT * mHat / (math.Sqrt(vHat) + o.eps)

		// Reset gradient
		p.Grad = 0
	}
}

// ============================================================================
// Section 12: Training Loop
// ============================================================================

// train runs the training loop for numSteps iterations.
func train(numSteps int, docs []string, uchars []rune, BOS, _ int, stateDict StateDict, optimizer *adamOptimizer) {
	lenBOS := 2
	vocabIndex := buildVocabIndex(uchars)

	for step := range numSteps {
		// Select document (round-robin)
		doc := docs[step%len(docs)]
		tokens := encode(doc, vocabIndex)

		// Wrap with BOS tokens
		allTokens := make([]int, 0, len(tokens)+lenBOS)
		allTokens = append(allTokens, BOS)
		allTokens = append(allTokens, tokens...)
		allTokens = append(allTokens, BOS)

		// Limit to blockSize
		n := min(len(allTokens)-1, blockSize)

		// Forward pass
		// Note: We build one full graph for the whole sequence (clear but not fast).
		// If you need speed, split the sequence into small chunks and backprop per chunk.
		keys := make([][][]*Value, nLayer)

		values := make([][][]*Value, nLayer)
		for i := range nLayer {
			keys[i] = make([][]*Value, 0)
			values[i] = make([][]*Value, 0)
		}

		losses := make([]*Value, 0, n)

		for posID := range n {
			tokenID := allTokens[posID]
			targetID := allTokens[posID+1]

			logits := gpt(tokenID, posID, keys, values, stateDict)
			probs := softmax(logits)

			// Cross-entropy loss: -log(probs[targetID])
			// Note: This keeps the softmax graph for clarity; log-sum-exp is a faster alternative.
			loss := neg(probs[targetID].log())
			losses = append(losses, loss)
		}

		// Average loss: (1/n) * sum(losses)
		// CRITICAL: Avoid integer division
		// Sum all losses first, then average (multiply by 1/n once)
		loss := losses[0] // Semantically Safe: n >= 1 guaranteed (allTokens has min 2 BOS tokens)
		for i := 1; i < len(losses); i++ {
			loss = add(loss, losses[i])
		}

		loss = mul(loss, newValue(1.0/float64(n), nil, nil))

		// Backward pass
		loss.backward()

		// Adam update
		optimizer.step(step)

		// Log progress (overwrite same line)
		fmt.Printf("\rstep %4d / %4d | loss %.4f", step+1, numSteps, loss.Data)
	}

	fmt.Println()
}

// ============================================================================
// Section 13: Weighted Random Sampling
// ============================================================================

// weightedChoice selects an index based on weights (normalized probabilities).
func weightedChoice(weights []float64) int {
	// Calculate cumulative sum
	cumsum := make([]float64, len(weights))

	cumsum[0] = weights[0]
	for i := 1; i < len(weights); i++ {
		cumsum[i] = cumsum[i-1] + weights[i]
	}

	// Generate random value in [0, total)
	total := cumsum[len(cumsum)-1]
	if total <= 0 {
		// Handle edge case: all weights are zero or negative
		return 0
	}

	r := rng.Float64() * total

	// Binary search or linear scan to find selected index
	for i, cs := range cumsum {
		if r < cs {
			return i
		}
	}

	return len(weights) - 1
}

// ============================================================================
// Section 13.5: Sampling
// ============================================================================

// sample generates a single sample from the model.
func sample(temperature float64, maxLen int, uchars []rune, BOS, _ int, stateDict StateDict) string {
	keys := make([][][]*Value, nLayer)

	values := make([][][]*Value, nLayer)
	for i := range nLayer {
		keys[i] = make([][]*Value, 0)
		values[i] = make([][]*Value, 0)
	}

	tokenID := BOS

	var result strings.Builder

	for posID := range maxLen {
		logits := gpt(tokenID, posID, keys, values, stateDict)

		// Temperature scaling: divide each logit by temperature
		scaledLogits := make([]*Value, len(logits))
		for i, l := range logits {
			scaledLogits[i] = mul(l, newValue(1.0/temperature, nil, nil))
		}

		// Softmax
		probs := softmax(scaledLogits)

		// Extract probabilities as float64 for weighted choice
		weights := make([]float64, len(probs))
		for i, p := range probs {
			weights[i] = p.Data
		}

		// Sample token
		tokenID = weightedChoice(weights)

		// Stop on BOS
		if tokenID == BOS {
			break
		}

		// Append character to result
		if tokenID < len(uchars) {
			result.WriteRune(uchars[tokenID])
		}
	}

	return result.String()
}
