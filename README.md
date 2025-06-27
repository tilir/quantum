# Quantum demos and experiments

Just different demos

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
- `inv test` — запуск тестов
- `inv run-experiments` — выполнение экспериментов
- `inv freeze` — обновление requirements.txt
