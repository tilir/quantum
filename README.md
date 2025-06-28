# Quantum demos and experiments

A collection of quantum computing demos and experiments.

## Installation

### 1. Create virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate     # Windows
```

### 2. Install dependencies
```bash
pip install invoke
inv install
```

### 3. Basic tasks
- `inv test` —  run tests
- `inv run-experiments` — run experiments
- `inv freeze` —  update `requirements.txt`
- `inv fix` — automatically format and lint the code
