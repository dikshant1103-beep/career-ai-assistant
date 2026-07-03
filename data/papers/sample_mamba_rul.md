# MambaRUL — Battery Remaining Useful Life via Mamba State-Space Models

## Abstract
We adapt the Mamba selective state-space architecture for the task of battery
remaining-useful-life (RUL) prediction. The architecture's linear-time
sequence modelling is well-matched to long charge/discharge histories where
LSTM and Transformer baselines either over-smooth or scale poorly.

## Contributions
1. A Mamba-based RUL head with PyBaMM-augmented training data.
2. Ablations on selective scan vs. SSM blocks vs. Transformer baselines.
3. Empirical results showing reduced MAE and lower variance on long-horizon RUL.

## Method
- Input: time-series of voltage, current, temperature, capacity per cycle.
- Backbone: Mamba layers with selective state-space scans.
- Output: scalar RUL in cycles + uncertainty estimate.

## Results
- ~14% lower MAE vs. LSTM baseline on the NASA dataset.
- 4× faster inference than a comparable Transformer.

## Future work
Couple with physics constraints from PyBaMM single-particle model for
physics-informed RUL.
