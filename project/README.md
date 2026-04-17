# ColPali Multi-Vector Optimization Project

A hands-on 10-minute project to learn multi-vector search optimization using **Scalar Quantization** or **MUVERA** techniques.

## Project Overview

**Goal**: Optimize ColPali multi-vector image retrieval by implementing ONE optimization technique and comparing its trade-offs.

**Time**: 10 minutes
**Difficulty**: Beginner (Quantization) / Advanced (MUVERA)
**Dataset**: 5 pages from "Attention is All You Need" paper

## What You'll Learn

- **Scalar Quantization**: Reduce memory by 75% with minimal accuracy loss
- **MUVERA**: Enable 10-20x faster search using HNSW approximate search
- **Trade-offs**: Understanding memory vs speed vs accuracy in vector search

## Project Structure

```
colpali-quick-optimization/
├── project.ipynb              # Main notebook (only 1 cell to fill!)
├── baseline_helper.py         # Helper module for baseline setup
├── spec.md                    # Project specification
├── docs.md                    # API reference for both approaches
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── attention_paper/
    ├── page-0.png            # Title page
    ├── page-1.png            # Abstract
    ├── page-3.png            # Architecture diagram
    ├── page-4.png            # Attention mechanism
    ├── page-8.png            # Results
    └── attention-embeddings-colsmol.parquet  # Pre-computed embeddings
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Open the Notebook

```bash
jupyter notebook project.ipynb
```

### 3. Follow the Instructions

1. **Cell 0**: Read the introduction
2. **Cell 1**: Understand the setup
3. **Cell 2**: Run baseline setup (see results immediately!)
4. **Cell 3**: Read the task and choose an optimization
5. **Cell 4**: Paste Jupyter AI generated code (THE ONLY CELL YOU FILL!)
6. **Cell 5**: Celebrate your results

## Optimization Options

### Option 1: Scalar Quantization (Recommended for Beginners)

**How it works**: Compresses float32 → int8 (4x compression)

**Trade-offs**:
- ✅ 75% memory reduction
- ✅ High accuracy (~99% precision)
- ⚠️ Similar speed to baseline

**Best for**: Memory-constrained environments, small to medium datasets

### Option 2: MUVERA (Advanced)

**How it works**: Converts multi-vectors to fixed-dimensional single vectors

**Trade-offs**:
- ✅ 10-20x faster search (enables HNSW)
- ⚠️ Lower accuracy (~40-60% precision)
- 💡 Best combined with reranking

**Best for**: Speed optimization, large-scale collections

## Files to Attach to Jupyter AI

When using Jupyter AI chatbot, attach these 3 files:

1. `project.ipynb` - Your notebook
2. `spec.md` - Project specification
3. `docs.md` - API documentation

## Example Queries

The baseline runs this query: `"transformer architecture diagram"`

**Expected result**: Page 3 (architecture diagram)

## Metrics Comparison

The helper module automatically compares:

- **Memory**: MB reduction
- **Search Time**: Speed improvement
- **Accuracy**: Top result match with baseline

## Production Tips

1. **Scalar Quantization**: Use INT8 for 4x compression, binary for 32x (lower accuracy)
2. **MUVERA**: Always combine with ColPali reranking for production
3. **Hybrid**: Use MUVERA for fast candidate retrieval, then rerank with ColPali

## Learn More

- [Qdrant Multi-Vector Documentation](https://qdrant.tech/documentation/concepts/vectors/#multivectors)
- [ColPali Paper](https://arxiv.org/abs/2407.01449)
- [MUVERA Paper](https://arxiv.org/abs/2410.10129)
- [JupyterAI Course](https://www.deeplearning.ai/short-courses/jupyter-ai-coding-in-notebooks/)

## Troubleshooting

**Import errors**: Make sure all dependencies from `requirements.txt` are installed

**Model download slow**: First run downloads ColPali model (~1GB), subsequent runs use cache

**MUVERA not found**: Install with `pip install muvera`

**Memory issues**: Use Scalar Quantization instead of MUVERA if low on RAM

## Credits

This project uses:
- **Qdrant**: Vector database
- **ColPali**: Multi-vector image retrieval model
- **MUVERA**: Multi-Vector Embedding Reduction Algorithm
- **Dataset**: "Attention is All You Need" (Vaswani et al., 2017)

## License

Educational use only. Dataset and model licenses apply.
