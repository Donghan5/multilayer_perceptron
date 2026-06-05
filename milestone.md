# To-Do list for test files

---

4 focused files, ~30 tests total — no trained_model fixture (too slow and brittle for a test suite):

tests/conftest.py — just the shared fixtures, no session-scoped training

tests/test_backprop.py — gradient check is the single most important test. If backprop is wrong, nothing works.

Numerical vs analytical gradient for weights and biases, both ReLU and Sigmoid
tests/test_stability.py — things that silently break numerics:

Softmax on extreme inputs (no NaN/Inf)
Cross-entropy at prediction boundaries (0.0, 1.0)
50-epoch training produces no NaN in loss/weights
tests/test_optimizer.py — verify the two solvers do what they claim:

Adam timestep increments correctly
Adam m/v shapes match weights after first update
SGD applies the correct update rule (weight actually decreases along gradient)
tests/test_io.py — the pipeline contract:

Save → load → predict gives identical outputs (uses network.forward() directly, bypassing the normalization in predict() to avoid the double-norm pitfall)
Argparse exposes all required CLI flags in main.py

---