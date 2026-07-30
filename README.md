# CMPT 310 Project

This project classifies Toronto housing projects as having low, medium, or
high return on investment (ROI) from economic and location features.

## Run the evaluation

```bash
python -m pip install -r requirements.txt
python -m Model.EvaluateModels --models dummy sklearn custom
```

The command performs reproducible 5-fold cross-validation on held-out real
projects and saves metrics, confusion matrices, and comparison plots in
`Artifacts/Milestone2`.

Run the checks with:

```bash
python -m unittest discover -s tests -v
```
