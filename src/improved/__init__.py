"""Improved DILI prediction pipeline.

Modules:
- data_utils: SMILES standardization, rich atom/bond features, scaffold split
- cv_utils: scaffold-grouped stratified k-fold + class-distribution parity report
- models: GCN / GAT / GraphSAGE / GIN / MPNN with explicit layer-level upgrades
- train_utils: class-weighted training loop with early stopping and LR scheduling
- stats_utils: DeLong / Wilcoxon / McNemar / bootstrap-CI for model comparison
- tune: Optuna (TPE) + Grid Search hyperparameter tuning per model
"""
