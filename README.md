# Environment

Python 3.13 is recommended.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

# Dependencies

The required packages and tested versions are listed in `requirements.txt`.

# Run

```bash
python -m pytest -q
python experiments/independent_context/run_main_experiment.py --formal --verify-determinism --output-dir outputs/independent_context
python experiments/independent_context/run_critic_ablation.py --output-dir outputs/critic_isolation --verify-determinism
python experiments/shared_representation/run_experiment.py --formal --verify-determinism --output-dir outputs/shared_representation
python experiments/coefficient_sensitivity/run_experiment.py --formal --verify-determinism --output-dir outputs/coefficient_sensitivity
python experiments/lob_contract/run_experiment.py --formal --verify-determinism --output-dir outputs/lob_contract
python scripts/verify_release.py
```
