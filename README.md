# Environment

Python 3.13 is recommended.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The adaptive ABIDES assay additionally requires Docker with `linux/amd64` support. Its frozen config names the required image, official IPPO commit, and base-checkpoint hashes.

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
python revision/experiments/setup_official_on_policy.py
python -m pytest -q revision/tests/test_abides_typed_audit_ippo.py revision/tests/test_abides_typed_audit_runner.py
python scripts/verify_release.py
```

With the pinned Docker image and base checkpoints from `revision/configs/abides_typed_audit_ippo_v2.json` available, run the adaptive assay with:

```bash
python revision/experiments/run_abides_typed_audit_ippo_v1.py --config revision/configs/abides_typed_audit_ippo_v2.json --engineering-smoke
python revision/experiments/run_abides_typed_audit_ippo_v1.py --config revision/configs/abides_typed_audit_ippo_v2.json --formal
```
