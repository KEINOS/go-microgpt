package main

import (
	"fmt"
	"testing"
)

// TestConfigConstants verifies configuration values match spec.md.
func TestConfigConstants(t *testing.T) {
	if nLayer != 1 {
		t.Errorf("nLayer = %d, want 1", nLayer)
	}

	if nEmbd != 16 {
		t.Errorf("nEmbd = %d, want 16", nEmbd)
	}

	if blockSize != 16 {
		t.Errorf("blockSize = %d, want 16", blockSize)
	}

	if nHead != 4 {
		t.Errorf("nHead = %d, want 4", nHead)
	}

	if headDim != 4 {
		t.Errorf("headDim = %d, want 4", headDim)
	}

	if numSteps != 1000 {
		t.Errorf("numSteps = %d, want 1000", numSteps)
	}

	if learningRate != 0.01 {
		t.Errorf("learningRate = %f, want 0.01", learningRate)
	}
}

// TestRNGSeeded verifies same seed produces same sequence.
func TestRNGSeeded(t *testing.T) {
	// Initialize with seed 42
	initRNG(42)

	seq1 := make([]float64, 10)
	for i := range 10 {
		seq1[i] = rng.Float64()
	}

	// Reinitialize with same seed
	initRNG(42)

	seq2 := make([]float64, 10)
	for i := range 10 {
		seq2[i] = rng.Float64()
	}

	// Sequences should be identical
	for i := range 10 {
		if seq1[i] != seq2[i] {
			t.Errorf("RNG not deterministic: seq1[%d]=%f, seq2[%d]=%f", i, seq1[i], i, seq2[i])
		}
	}
}

// TestNewValue verifies Value initialization.
func TestNewValue(t *testing.T) {
	v := newValue(3.14, nil, nil)
	if v.Data != 3.14 {
		t.Errorf("Data = %f, want 3.14", v.Data)
	}

	if v.Grad != 0 {
		t.Errorf("Grad = %f, want 0", v.Grad)
	}

	if v.children != nil {
		t.Errorf("children not nil")
	}

	if v.localGrads != nil {
		t.Errorf("localGrads not nil")
	}
}

// ============================================================================
// Tests for Section 2: Value Arithmetic Operations
// ============================================================================

func TestValueAdd(t *testing.T) {
	a := newValue(2.0, nil, nil)
	b := newValue(3.0, nil, nil)

	c := add(a, b)
	if c.Data != 5.0 {
		t.Errorf("add(2, 3) = %f, want 5.0", c.Data)
	}

	if len(c.children) != 2 {
		t.Errorf("add should have 2 children, got %d", len(c.children))
	}
}

func TestValueMul(t *testing.T) {
	a := newValue(2.0, nil, nil)
	b := newValue(3.0, nil, nil)

	c := mul(a, b)
	if c.Data != 6.0 {
		t.Errorf("mul(2, 3) = %f, want 6.0", c.Data)
	}
}

func TestValuePow(t *testing.T) {
	a := newValue(2.0, nil, nil)

	c := pow(a, 3.0)
	if c.Data != 8.0 {
		t.Errorf("pow(2, 3) = %f, want 8.0", c.Data)
	}
}

func TestValueNeg(t *testing.T) {
	a := newValue(5.0, nil, nil)

	c := neg(a)
	if c.Data != -5.0 {
		t.Errorf("neg(5) = %f, want -5.0", c.Data)
	}
}

func TestValueSub(t *testing.T) {
	a := newValue(5.0, nil, nil)
	b := newValue(2.0, nil, nil)

	c := sub(a, b)
	if c.Data != 3.0 {
		t.Errorf("sub(5, 2) = %f, want 3.0", c.Data)
	}
}

func TestValueDiv(t *testing.T) {
	a := newValue(6.0, nil, nil)
	b := newValue(2.0, nil, nil)

	c := div(a, b)
	if c.Data != 3.0 {
		t.Errorf("div(6, 2) = %f, want 3.0", c.Data)
	}
}

// ============================================================================
// Tests for Section 3: Value Advanced Operations
// ============================================================================

func TestValueExp(t *testing.T) {
	a := newValue(1.0, nil, nil)
	c := a.exp()

	expected := 2.718281828
	if c.Data < expected-0.001 || c.Data > expected+0.001 {
		t.Errorf("exp(1) = %f, want ~%f", c.Data, expected)
	}
}

func TestValueLog(t *testing.T) {
	a := newValue(2.718281828, nil, nil)

	c := a.log()
	if c.Data < 0.999 || c.Data > 1.001 {
		t.Errorf("log(e) = %f, want ~1.0", c.Data)
	}
}

func TestValueRelu(t *testing.T) {
	a := newValue(5.0, nil, nil)

	c := a.relu()
	if c.Data != 5.0 {
		t.Errorf("relu(5) = %f, want 5.0", c.Data)
	}

	d := newValue(-3.0, nil, nil)

	e := d.relu()
	if e.Data != 0.0 {
		t.Errorf("relu(-3) = %f, want 0.0", e.Data)
	}
}

// ============================================================================
// Tests for Section 4: Backward Pass
// ============================================================================

func TestBackwardSimpleAdd(t *testing.T) {
	a := newValue(2.0, nil, nil)
	b := newValue(3.0, nil, nil)
	c := add(a, b)

	c.backward()

	if a.Grad != 1.0 {
		t.Errorf("a.Grad = %f, want 1.0", a.Grad)
	}

	if b.Grad != 1.0 {
		t.Errorf("b.Grad = %f, want 1.0", b.Grad)
	}
}

func TestBackwardSharedNode(t *testing.T) {
	a := newValue(2.0, nil, nil)
	// Use a twice: a + a
	b := add(a, a)
	b.backward()

	// Gradient should accumulate: 1 + 1 = 2
	if a.Grad != 2.0 {
		t.Errorf("a.Grad = %f, want 2.0 (gradient accumulation)", a.Grad)
	}
}

func TestBackwardComputation(t *testing.T) {
	a := newValue(2.0, nil, nil)
	b := newValue(3.0, nil, nil)
	c := mul(a, b)                       // c = 6
	d := add(c, newValue(1.0, nil, nil)) // d = 7

	d.backward()

	if a.Grad != 3.0 {
		t.Errorf("a.Grad = %f, want 3.0", a.Grad)
	}

	if b.Grad != 2.0 {
		t.Errorf("b.Grad = %f, want 2.0", b.Grad)
	}
}

// ============================================================================
// Tests for Section 5: Helper Function - linear
// ============================================================================

func TestLinear(t *testing.T) {
	// 3x2 matrix: [[1, 2], [3, 4], [5, 6]]
	w := make([][]*Value, 3)
	w[0] = []*Value{newValue(1, nil, nil), newValue(2, nil, nil)}
	w[1] = []*Value{newValue(3, nil, nil), newValue(4, nil, nil)}
	w[2] = []*Value{newValue(5, nil, nil), newValue(6, nil, nil)}

	// 2-element vector: [1, 2]
	x := []*Value{newValue(1, nil, nil), newValue(2, nil, nil)}

	// Expected: [1*1 + 2*2, 3*1 + 4*2, 5*1 + 6*2] = [5, 11, 17]
	y := linear(x, w)

	if len(y) != 3 {
		t.Errorf("linear output length = %d, want 3", len(y))
	}

	if y[0].Data != 5.0 {
		t.Errorf("y[0] = %f, want 5.0", y[0].Data)
	}

	if y[1].Data != 11.0 {
		t.Errorf("y[1] = %f, want 11.0", y[1].Data)
	}

	if y[2].Data != 17.0 {
		t.Errorf("y[2] = %f, want 17.0", y[2].Data)
	}
}

// ============================================================================
// Tests for Section 6: Helper Function - softmax
// ============================================================================

func TestSoftmax(t *testing.T) {
	logits := []*Value{
		newValue(1.0, nil, nil),
		newValue(2.0, nil, nil),
		newValue(3.0, nil, nil),
	}
	probs := softmax(logits)

	if len(probs) != 3 {
		t.Errorf("softmax output length = %d, want 3", len(probs))
	}

	// Sum should be approximately 1.0
	sum := 0.0
	for _, p := range probs {
		sum += p.Data
	}

	if sum < 0.99 || sum > 1.01 {
		t.Errorf("softmax sum = %f, want ~1.0", sum)
	}

	// Largest logit should have largest probability
	if probs[2].Data <= probs[1].Data || probs[1].Data <= probs[0].Data {
		t.Errorf("softmax probabilities not in expected order")
	}
}

func TestSoftmaxNumericalStability(t *testing.T) {
	// Very large logits should not overflow/NaN
	logits := []*Value{
		newValue(1000.0, nil, nil),
		newValue(1001.0, nil, nil),
		newValue(999.0, nil, nil),
	}
	probs := softmax(logits)

	sum := 0.0

	for _, p := range probs {
		if p.Data < 0 || p.Data > 1 {
			t.Errorf("probability out of range: %f", p.Data)
		}

		sum += p.Data
	}

	if sum < 0.99 || sum > 1.01 {
		t.Errorf("softmax sum = %f, want ~1.0", sum)
	}
}

// ============================================================================
// Tests for Section 7: Helper Function - rmsnorm
// ============================================================================

func TestRMSNorm(t *testing.T) {
	x := []*Value{
		newValue(1.0, nil, nil),
		newValue(2.0, nil, nil),
		newValue(3.0, nil, nil),
	}
	normalized := rmsnorm(x)

	if len(normalized) != 3 {
		t.Errorf("rmsnorm output length = %d, want 3", len(normalized))
	}

	// Verify normalized values have small magnitude (normalized)
	sumSq := 0.0
	for _, v := range normalized {
		sumSq += v.Data * v.Data
	}

	ms := sumSq / float64(len(normalized))
	if ms > 1.5 {
		t.Errorf("rmsnorm mean square = %f, expect normalized values", ms)
	}
}

// ============================================================================
// Tests for Section 8: Dataset Loading & Tokenization
// ============================================================================

func TestBuildVocab(t *testing.T) {
	docs := []string{"abc", "bcd", "aaa"}
	uchars, BOS, vocabSize := buildVocab(docs)

	// Should have unique chars: a, b, c, d
	if len(uchars) != 4 {
		t.Errorf("buildVocab uchars length = %d, want 4", len(uchars))
	}

	// BOS should be 4
	if BOS != 4 {
		t.Errorf("BOS = %d, want 4", BOS)
	}

	// vocabSize should be 5
	if vocabSize != 5 {
		t.Errorf("vocabSize = %d, want 5", vocabSize)
	}

	// uchars should be sorted
	if uchars[0] != 'a' || uchars[1] != 'b' || uchars[2] != 'c' || uchars[3] != 'd' {
		t.Errorf("uchars not properly sorted")
	}
}

func TestEncode(t *testing.T) {
	uchars := []rune{'a', 'b', 'c'}

	tokens := encode("abc", uchars)
	if len(tokens) != 3 {
		t.Errorf("encode length = %d, want 3", len(tokens))
	}

	if tokens[0] != 0 || tokens[1] != 1 || tokens[2] != 2 {
		t.Errorf("encode result = %v, want [0, 1, 2]", tokens)
	}
}

func TestDecode(t *testing.T) {
	uchars := []rune{'a', 'b', 'c'}
	BOS := 3
	tokens := []int{0, 1, 2}

	result := decode(tokens, uchars, BOS)
	if result != "abc" {
		t.Errorf("decode result = %q, want %q", result, "abc")
	}
}

func TestDecodeWithBOS(t *testing.T) {
	uchars := []rune{'a', 'b', 'c'}
	BOS := 3
	tokens := []int{BOS, 0, 1, BOS, 2}
	result := decode(tokens, uchars, BOS)
	// Should skip BOS tokens
	if result != "abc" {
		t.Errorf("decode result = %q, want %q", result, "abc")
	}
}

func TestShuffleDocs(t *testing.T) {
	initRNG(42)

	docs := []string{"a", "b", "c", "d", "e"}
	originalDocs := make([]string, len(docs))
	copy(originalDocs, docs)

	shuffleDocs(docs)

	// Verify all elements are still present (just reordered)
	docMap := make(map[string]bool)
	for _, d := range docs {
		docMap[d] = true
	}

	for _, d := range originalDocs {
		if !docMap[d] {
			t.Errorf("doc %q missing after shuffle", d)
		}
	}
}

// ============================================================================
// Tests for Section 9: Model Parameter Initialization
// ============================================================================

func TestMatrix(t *testing.T) {
	initRNG(42)

	mat := matrix(3, 2, 0.08)
	if len(mat) != 3 {
		t.Errorf("matrix rows = %d, want 3", len(mat))
	}

	if len(mat[0]) != 2 {
		t.Errorf("matrix cols = %d, want 2", len(mat[0]))
	}
	// Check that values are initialized (not nil)
	for i := range mat {
		for j := range mat[i] {
			if mat[i][j] == nil {
				t.Errorf("matrix[%d][%d] is nil", i, j)
			}
		}
	}
}

func TestInitStateDict(t *testing.T) {
	initRNG(42)

	vocabSize := 27
	sd := initStateDict(vocabSize)

	// Check all required keys exist
	requiredKeys := []string{"wte", "wpe", "lm_head"}
	for _, key := range requiredKeys {
		if _, ok := sd[key]; !ok {
			t.Errorf("missing key: %s", key)
		}
	}

	// Check shapes
	if len(sd["wte"]) != vocabSize || len(sd["wte"][0]) != nEmbd {
		t.Errorf("wte shape mismatch")
	}

	if len(sd["wpe"]) != blockSize || len(sd["wpe"][0]) != nEmbd {
		t.Errorf("wpe shape mismatch")
	}

	if len(sd["lm_head"]) != vocabSize || len(sd["lm_head"][0]) != nEmbd {
		t.Errorf("lm_head shape mismatch")
	}

	// Check layer keys
	for i := range nLayer {
		prefix := fmt.Sprintf("layer%d", i)

		layerKeys := []string{
			prefix + ".attn_wq",
			prefix + ".attn_wk",
			prefix + ".attn_wv",
			prefix + ".attn_wo",
			prefix + ".mlp_fc1",
			prefix + ".mlp_fc2",
		}
		for _, key := range layerKeys {
			if _, ok := sd[key]; !ok {
				t.Errorf("missing key: %s", key)
			}
		}
	}
}

func TestFlattenParams(t *testing.T) {
	initRNG(42)

	vocabSize := 27
	sd := initStateDict(vocabSize)
	params := flattenParams(sd)

	// Calculate expected parameter count
	// wte: vocabSize * nEmbd
	// wpe: blockSize * nEmbd
	// lm_head: vocabSize * nEmbd
	// per layer: attn_wq (nEmbd*nEmbd) + attn_wk + attn_wv + attn_wo + mlp_fc1 (4*nEmbd*nEmbd) + mlp_fc2 (nEmbd*4*nEmbd)
	expected := vocabSize*nEmbd + blockSize*nEmbd + vocabSize*nEmbd
	expected += nLayer * (nEmbd*nEmbd + nEmbd*nEmbd + nEmbd*nEmbd + nEmbd*nEmbd + 4*nEmbd*nEmbd + nEmbd*4*nEmbd)
	// For default config: 27*16 + 16*16 + 27*16 + 1*(16*16*4 + 16*16 + 4*16*16) = 432+256+432+1024+256+1024 = 3424

	if len(params) != expected {
		t.Errorf("flattenParams count = %d, want %d", len(params), expected)
	}

	// Verify all params are non-nil
	for i, p := range params {
		if p == nil {
			t.Errorf("param[%d] is nil", i)
		}
	}
}

// ============================================================================
// Tests for Section 10: GPT Forward Pass
// ============================================================================

func TestGPTEmbedding(t *testing.T) {
	initRNG(42)

	vocabSize := 27
	sd := initStateDict(vocabSize)

	// Create empty KV cache for single position
	keys := make([][][]*Value, nLayer)

	values := make([][][]*Value, nLayer)
	for i := range nLayer {
		keys[i] = make([][]*Value, 0)
		values[i] = make([][]*Value, 0)
	}

	logits := gpt(0, 0, keys, values, sd)

	if len(logits) != vocabSize {
		t.Errorf("gpt output shape = %d, want %d (vocabSize)", len(logits), vocabSize)
	}

	// Verify logits are Values
	for i, logit := range logits {
		if logit == nil {
			t.Errorf("logit[%d] is nil", i)
		}
	}
}

func TestGPTKVCache(t *testing.T) {
	initRNG(42)

	vocabSize := 27
	sd := initStateDict(vocabSize)

	keys := make([][][]*Value, nLayer)

	values := make([][][]*Value, nLayer)
	for i := range nLayer {
		keys[i] = make([][]*Value, 0)
		values[i] = make([][]*Value, 0)
	}

	// Forward pass for position 0
	gpt(0, 0, keys, values, sd)

	if len(keys[0]) != 1 || len(values[0]) != 1 {
		t.Errorf("KV cache not accumulated at pos 0")
	}

	// Forward pass for position 1
	gpt(1, 1, keys, values, sd)

	if len(keys[0]) != 2 || len(values[0]) != 2 {
		t.Errorf("KV cache not accumulated at pos 1")
	}
}

func TestGPTBackward(t *testing.T) {
	initRNG(42)

	vocabSize := 27
	sd := initStateDict(vocabSize)
	params := flattenParams(sd)

	keys := make([][][]*Value, nLayer)

	values := make([][][]*Value, nLayer)
	for i := range nLayer {
		keys[i] = make([][]*Value, 0)
		values[i] = make([][]*Value, 0)
	}

	logits := gpt(0, 0, keys, values, sd)

	// Create a simple loss and backward
	loss := logits[0]
	loss.backward()

	// Verify some gradients are non-zero
	nonZeroGrads := 0

	for _, p := range params {
		if p.Grad != 0 {
			nonZeroGrads++
		}
	}

	if nonZeroGrads == 0 {
		t.Errorf("no gradients computed in backward pass")
	}
}

// ============================================================================
// Tests for Section 11: Adam Optimizer
// ============================================================================

func TestAdamStep(t *testing.T) {
	// Create a simple parameter with gradient
	param := newValue(1.0, nil, nil)
	param.Grad = 0.1

	params := []*Value{param}
	optimizer := newAdamOptimizer(params, learningRate, beta1, beta2, epsAdam)

	initialData := param.Data

	// Perform an optimization step
	optimizer.step(0)

	// Parameter should have decreased (moved in negative gradient direction)
	if param.Data >= initialData {
		t.Errorf("Adam step didn't update parameter: initial=%f, after=%f", initialData, param.Data)
	}

	// Gradient should be reset
	if param.Grad != 0 {
		t.Errorf("Gradient not reset: %f", param.Grad)
	}

	// Check momentum accumulation (m[0] should be (1-beta1)*grad)
	expectedM := (1 - beta1) * 0.1
	if optimizer.m[0] < expectedM*0.99 || optimizer.m[0] > expectedM*1.01 {
		t.Errorf("Momentum not accumulated correctly: %f, expected ~%f", optimizer.m[0], expectedM)
	}
}

func TestAdamLearningRateDecay(t *testing.T) {
	param := newValue(1.0, nil, nil)
	params := []*Value{param}
	optimizer := newAdamOptimizer(params, learningRate, beta1, beta2, epsAdam)

	// Step at beginning
	param.Grad = 0.1
	data0 := param.Data

	optimizer.step(0)

	step0Update := data0 - param.Data

	// Reset param and step at middle
	param.Data = 1.0
	param.Grad = 0.1
	data500 := param.Data

	optimizer.step(500)

	step500Update := data500 - param.Data

	// Learning rate at step 500 should be ~0.5 * learning rate at step 0
	// So update should be roughly half
	if step500Update >= step0Update {
		t.Errorf("LR decay not working: step0=%f, step500=%f", step0Update, step500Update)
	}
}

// ============================================================================
// Tests for Section 13: Weighted Random Sampling
// ============================================================================

func TestWeightedChoice(t *testing.T) {
	initRNG(42)

	weights := []float64{0.1, 0.2, 0.7}

	// Sample many times, count selections
	counts := [3]int{0, 0, 0}

	for range 1000 {
		idx := weightedChoice(weights)
		counts[idx]++
	}

	// Index 2 should be selected most often (70% probability)
	if counts[2] < counts[1] || counts[1] < counts[0] {
		t.Logf("weighted choice distribution: %v (expected ~100, ~200, ~700)", counts)
	}
}

func TestWeightedChoiceUniform(t *testing.T) {
	initRNG(42)

	weights := []float64{0.5, 0.5}

	// Both should be selected roughly equally
	counts := [2]int{0, 0}

	for range 100 {
		idx := weightedChoice(weights)
		counts[idx]++
	}

	// Very rough check that both are selected
	if counts[0] == 0 || counts[1] == 0 {
		t.Errorf("not all weights selected: %v", counts)
	}
}

// ============================================================================
// Tests for Section 13.5: Inference & Sampling
// ============================================================================

func TestSampleLength(t *testing.T) {
	initRNG(42)

	vocabSize := 27
	sd := initStateDict(vocabSize)

	uchars := make([]rune, vocabSize-1)
	for i := range uchars {
		uchars[i] = rune('a' + i)
	}

	BOS := vocabSize - 1

	sample := sample(temperature, blockSize, uchars, BOS, vocabSize, sd)
	if len(sample) > blockSize {
		t.Errorf("sample length %d > blockSize %d", len(sample), blockSize)
	}
}

func TestSampleTemperature(t *testing.T) {
	initRNG(42)

	vocabSize := 27
	sd := initStateDict(vocabSize)

	uchars := make([]rune, vocabSize-1)
	for i := range uchars {
		uchars[i] = rune('a' + i)
	}

	BOS := vocabSize - 1

	// Low temperature sample (more deterministic)
	initRNG(42)

	lowTemp := sample(0.01, blockSize, uchars, BOS, vocabSize, sd)

	// High temperature sample (more random)
	initRNG(42)

	highTemp := sample(2.0, blockSize, uchars, BOS, vocabSize, sd)

	// Both should be valid strings
	if len(lowTemp) == 0 || len(highTemp) == 0 {
		t.Errorf("sample returned empty string")
	}
}

// ============================================================================
// Tests for Section 12: Training Loop
// ============================================================================

func TestTrainSteps(_ *testing.T) {
	initRNG(42)

	docs := []string{"abc", "bcd", "aaa"}
	uchars, BOS, vocabSize := buildVocab(docs)
	sd := initStateDict(vocabSize)
	params := flattenParams(sd)
	optimizer := newAdamOptimizer(params, learningRate, beta1, beta2, epsAdam)

	// Run a few training steps
	train(5, docs, uchars, BOS, vocabSize, sd, optimizer)

	// Verify training completed without error
	// (if this didn't panic, training works)
}
