# Skip-Gram Variant Evaluation

## Overview
This project provides a systematic empirical evaluation of distributional word representations learned using the Skip-Gram model. Multiple Skip-Gram variants are compared, including **Hierarchical Softmax** and **Negative Sampling**, each evaluated **with and without subsampling of frequent words**, against a **TF-IDF baseline**.

The analysis focuses on semantic quality, embedding geometry, clustering behavior, and computational efficiency under controlled preprocessing and a shared vocabulary. Intrinsic evaluation benchmarks are used to understand how training objectives and frequency-based regularization affect learned word representations.

---

## Table of Contents
1. [Objective](#objective)
2. [Data Sources](#data-sources)
3. [Methodology](#methodology)
4. [Code Structure](#code-structure)
5. [Results](#results)
6. [Evaluation and Analysis](#evaluation-and-analysis)
7. [Future Improvements](#future-improvements)

---

## Order of Operations
1. Corpus construction by merging Simple English Wikipedia and news headlines.
2. Text normalization, cleaning, sentence segmentation, and tokenization.
3. Vocabulary construction with frequency thresholds and shared vocabulary.
4. Training Skip-Gram models using:
   - Hierarchical Softmax (with and without subsampling)
   - Negative Sampling (with and without subsampling)
5. Training a TF-IDF baseline using the same vocabulary.
6. Evaluating models on semantic similarity, analogies, clustering, geometry, and efficiency.
7. Comparing performance, structure, and computational tradeoffs across models.

---

## Objective
The primary goals of this project are to:
- Compare Skip-Gram word embeddings against a TF-IDF baseline under controlled conditions.
- Evaluate the impact of training objective (Hierarchical Softmax vs. Negative Sampling).
- Analyze the role of subsampling in semantic quality and efficiency.
- Study embedding geometry, clustering behavior, and frequency effects.
- Understand tradeoffs between representational quality and computational cost.

---

## Data Sources
The project uses a merged text corpus composed of:
1. **Simple English Wikipedia**: Long-form explanatory text.
2. **A Million News Headlines**: Short, information-dense sentences.

Both datasets are publicly available and combined into a single corpus to ensure consistent exposure across all models.

---

## Methodology
### 1. Data Preprocessing
- Unicode normalization and lowercasing.
- Removal of URLs, emails, bracketed content, and extraneous punctuation.
- Sentence segmentation using punctuation and line breaks.
- Tokenization preserving internal apostrophes and hyphens.
- Fallback chunking for sparse sentence boundaries.

### 2. Vocabulary Construction
- Minimum frequency threshold: `min_count = 10`.
- Maximum vocabulary size: `100,000`.
- A single shared vocabulary is used across all models.

### 3. Model Variants
The following models are trained and evaluated:
- **Skip-Gram + Hierarchical Softmax**
- **Skip-Gram + Negative Sampling (K = 10)**
- Each Skip-Gram variant is evaluated:
  - With subsampling (`t = 1e-4`)
  - Without subsampling
- **TF-IDF baseline** with L2 normalization

All Skip-Gram models use:
- Embedding dimension: 200  
- Dynamic context window (max radius = 5)  
- Sparse updates and PyTorch implementation  

### 4. Evaluation Metrics
- **Semantic similarity**: WordSim-353 (Spearman correlation)
- **Analogy reasoning**: Google Word Analogy dataset (top-1 accuracy)
- **Clustering quality**: NMI and ARI over semantic categories
- **Embedding geometry**: PCA, t-SNE, norm–frequency regression, hubness
- **Efficiency**: Runtime, throughput, CPU and memory usage

---

## Code Structure
The project is organized into the following components:

### 1. Data Processing
- Corpus merging and normalization.
- Sentence segmentation and tokenization.
- Vocabulary construction and frequency statistics.

### 2. Skip-Gram Implementation
- Dynamic window pair generation.
- Hierarchical Softmax using Huffman trees.
- Negative Sampling with unigram noise distribution.
- Optional subsampling of frequent words.

### 3. Training and Monitoring
- Mini-batch training with validation split.
- Learning rate scheduling and early stopping.
- CPU, memory, and runtime monitoring during training.

### 4. Evaluation and Analysis
- Intrinsic semantic benchmarks.
- Clustering and geometric analysis.
- Visualization using PCA and t-SNE.
- Comparative analysis across model variants.

---

## Results
### Key Findings
1. **Semantic Performance**
   - Skip-Gram embeddings consistently outperform TF-IDF on similarity and analogy tasks.
   - Negative Sampling with subsampling achieves the strongest overall performance.

2. **Clustering and Structure**
   - Skip-Gram embeddings form well-separated semantic clusters.
   - TF-IDF shows weak global semantic organization.

3. **Embedding Geometry**
   - TF-IDF vectors are dominated by word frequency.
   - Subsampling reduces frequency-driven distortion in Skip-Gram embeddings.
   - Negative Sampling produces healthier neighborhood structures.

4. **Computational Efficiency**
   - Negative Sampling is significantly faster than Hierarchical Softmax.
   - Subsampling substantially reduces training time and memory usage.
   - Hierarchical Softmax without subsampling is the most computationally expensive configuration.

---

## Evaluation and Analysis
The results demonstrate that predictive embedding learning captures semantic relationships more effectively than count-based methods. Subsampling acts not only as an efficiency mechanism but also as a form of geometric regularization, improving both representation quality and stability. Among all configurations, **Skip-Gram with Negative Sampling and subsampling** provides the best balance between accuracy, structure, and efficiency.

---

## Future Improvements
1. **Downstream Task Evaluation**
   - Evaluate embeddings on classification or sequence modeling tasks.

2. **Larger or Multilingual Corpora**
   - Extend analysis to larger or multilingual datasets.

3. **Contextual Embeddings**
   - Compare results with contextual models such as BERT or GloVe.

4. **Hyperparameter Sensitivity**
   - Explore different embedding dimensions, window sizes, and negative sample counts.

5. **Visualization Extensions**
   - Add interactive visualizations for embedding exploration.

---

## Conclusion
This project provides a controlled empirical comparison of Skip-Gram word embeddings and a TF-IDF baseline. The results show that Skip-Gram consistently produces richer semantic representations, with **Negative Sampling combined with subsampling** emerging as the most effective and efficient configuration. The findings highlight how training objectives and frequency-based regularization jointly shape representation quality, efficiency, and embedding geometry.

---

## Acknowledgments
- **Libraries Used**: NumPy, Pandas, SciPy, Scikit-learn, PyTorch, Matplotlib
- **Datasets**: Simple English Wikipedia, A Million News Headlines
- **References**: Mikolov et al. (2013), WordSim-353, Google Word Analogy Dataset