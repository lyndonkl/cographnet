# CoGraphNet Implementation

A PyTorch Geometric implementation of CoGraphNet for text classification using word-sentence heterogeneous graph representations.

## Description

This project implements the CoGraphNet architecture as described in the paper "CoGraphNet for enhanced text classification using word-sentence heterogeneous graph representations and improved interpretability". The implementation uses PyTorch Geometric for graph neural networks and BERT for text embeddings.

## Project Structure

```
cographnet/
├── README.md
├── environment.yml
├── src/
│   ├── data/
│   │   └── graph_builder.py
│   ├── models/
│   └── utils/
├── tests/
└── examples/
```

## Installation

For M1/M2/M3 Macs:
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate cographnet

# Install PyG using pip (required for Apple Silicon)
pip install torch-geometric
```

## Usage

```python
from pathlib import Path
from src.data.graph_builder import GraphBuilder

# Initialize graph builder
builder = GraphBuilder(window_size=3, alpha=0.5)

# Process documents
json_files = list(Path('data/').glob('*.json'))
graphs = builder.process_documents(json_files)
```

## Dependencies

- Python 3.12
- PyTorch 2.5.*
- PyTorch Geometric
- NumPy
- tqdm

## License

MIT License 