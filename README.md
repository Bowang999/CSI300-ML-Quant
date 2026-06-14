# CSI300-ML-Quant

Machine learning-based quantitative stock selection research on the CSI 300 (沪深300) index.

---

## Repository Contents

```
Figure/                     Backtest performance charts and model diagnostic plots
weights_diagnostic/
  paper_use/                Final model portfolio weights and IC diagnostics
CSI300_CHANGE_WIND/         CSI300 constituent lists (2010–2024)
```

---

## Models

Results are provided for six strategies, evaluated on monthly rebalancing frequency:

| Model | Description |
|---|---|
| Ridge | Ridge regression with MVO portfolio construction |
| XGBoost | Gradient boosting trees |
| LightGBM | LightGBM |
| LSTM + VSN | Long Short-Term Memory with Variable Selection Network |
| C-ENet | Cross-sectional ensemble |
| Equal-Weight Ensemble | Equal-weighted combination of all models |

---

## Data

Raw data is sourced from **CSMAR** and **Wind** (commercial licences required) and is not included in this repository.

CSI300 constituent membership files (2010–2024) are included under `CSI300_CHANGE_WIND/`.
