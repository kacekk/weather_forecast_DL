# weather_forecast_DL

Final project for the Deep Learning course (Koźmiński University 2026/27, [`pamelon/DeepLearning-ALK`](https://github.com/pamelon/DeepLearning-ALK), `Grading.md`, option 3 – Weather Dataset).

**Goal:** from the last 3 days of weather, predict the next day's temperature (regression). `Apparent Temperature (C)` is never used.
**Our definition of the target:** the next day's **daily mean** temperature (local time).

## Result (unseen test year 2016)

| Model | Test MAE (°C) |
|---|---|
| Climatology (average for that day of year) | 2.535 |
| Persistence (tomorrow = today) | 1.428 |
| **Final model** – mean ± std over 3 seeds | **1.269 ± 0.048** |
| **Final model** – average of the 3 seeds' predictions | **1.232** |

That's 11–14 % better than persistence. 5-fold K-fold over years (2006–2015) gave an estimate of 1.156 ± 0.030 °C.

**Final network:** two GRU branches – the 72 hourly readings (14 features) and the 3 daily rows (17 features) – plus the day of year → `Dense(32, relu)` → `Dropout(0.4)` → `Dense(1)`. It predicts the **change** from today's mean. Huber loss, Adam with ReduceLROnPlateau.

## How it was built

The workflow follows the course materials:
- **Class 1 – reproducible experiments:** settings in `config.yaml`, fixed seeds (123, 2137, 15190), a split made once, normalisation fitted on training data only, every configuration trained with 3 seeds and reported as mean ± std, the test set kept for the end, and a reproducibility check.
- **Class 2 – tuning:** start from the tabular baseline (Dense 64 → 32 → 1, Adam, MSE), change one thing per experiment, add dropout when overfitting, tune learning rate and batch size.
- **`regression.ipynb`:** scale the target, use K-fold validation, retrain the final model on all non-test data.

Search: 46 configurations, 219 training runs (≈ 55 min on an Apple-silicon CPU). Every run is logged in `runs/experiments.jsonl`.

## Files

| File | What it is |
|---|---|
| `weather_forecast.ipynb` | **the report** – cleaning, exploration, features, experiments, final result (matches the presentation outline in `Grading.md`) |
| `weather_lib.py` | data cleaning, feature engineering, windowing, model building and training |
| `experiments.py` | the full search → K-fold → ablation → final test pipeline |
| `config.yaml` | every setting, including the search stages |
| `runs/` | experiment log, results tables (`search_steps.csv`, `cv.csv`, `ablation.csv`), final models, test predictions; `final_first_pass/` holds the abandoned first-pass final model |
| `requirements.txt` | exact library versions |

## Reproduce

The dataset is not included in this repo. Download it from Kaggle ([arnab132/weather-dataset](https://www.kaggle.com/datasets/arnab132/weather-dataset), file `weatherHistory.csv`) into `data/`. `config.yaml` stores its SHA-256 hash, and the code checks it on load.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data && python -c "import kagglehub, shutil; shutil.copy(kagglehub.dataset_download('arnab132/weather-dataset') + '/weatherHistory.csv', 'data/')"
python experiments.py              # full experiment pipeline (finished runs in runs/experiments.jsonl are reused)
jupyter lab weather_forecast.ipynb # the report
```

TensorFlow doesn't support Python 3.14 yet, so use Python 3.12.

## Use of AI tools

Required by the course: the code structure and first drafts of the explanations were produced with Claude Code (Anthropic) and reviewed by the team. Details are at the end of the notebook.
