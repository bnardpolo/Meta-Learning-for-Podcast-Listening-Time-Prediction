# podcast_meta_learning

**Experiment Date:** 2025-10-27 10:27:19

## Directory Structure

```
podcast_meta_learning_20251027_102719/
|-- models/           # Trained model files (.pkl)
|-- results/          # Metrics and results (.json, .csv)
|-- plots/            # Visualizations (.png)
|-- config.json       # Experiment configuration
`-- README.md         # This file
```

## Models Saved

- baseline_model
- improved_proto

## Usage

```python
import pickle

# Load a model
with open('models/baseline_model.pkl', 'rb') as f:
    model = pickle.load(f)
```
