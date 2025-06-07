# Quadratic Equation Solver Training

This repository contains a Python script that trains a small neural
network to approximate the solutions of randomly generated quadratic
equations.

The script `train_quadratic_solver.py` implements the network and
training logic using only Python's standard library. During training, it
repeatedly generates a random equation of the form `ax^2 + bx + c = 0`
with real roots, computes the exact solutions via the quadratic formula,
and performs one gradient-descent step. This process is repeated one
million times by default.

To run the training:

```bash
python train_quadratic_solver.py
```

Model parameters are saved to `quadratic_model.json` after training.
