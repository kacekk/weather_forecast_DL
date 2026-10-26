# weather_forecast_DL

Final project for the Deep Learning course (Koźmiński University 2026/27, [`pamelon/DeepLearning-ALK`](https://github.com/pamelon/DeepLearning-ALK), `Grading.md`, option 3 – Weather Dataset).

**Goal:** from the last 3 days of weather, predict the next day's temperature (regression). `Apparent Temperature (C)` is never used.
**Our definition of the target:** the next day's **daily mean** temperature (local time).

## Result (unseen test year 2016)

| Model | Test MAE (°C) |
|---|---|
| Climatology (average for that day of year) | 2.535 |
| Persistence (tomorrow = today) | 1.428 |
| Final network – single models (3 seeds, mean ± std) | 1.178 ± 0.010 |
| **Final network – average of the 3 seeds** | **1.160** |

That's **19 % better than persistence**. K-fold over 2006–2015 estimated 1.123 °C. On the same folds, gradient boosting reaches 1.221 °C and Ridge regression 1.288 °C.

**Final network:** two GRU branches (the 72 hourly readings, with attention pooling, and the 3 daily rows) plus the day of year → `Dense(32, relu)` → `Dropout(0.4)` → `Dense(24)`. It predicts the next day's 24 hourly temperatures as changes from today's mean, and the predicted daily mean is their average. Three models trained with different seeds are averaged. The notebook justifies every component with a K-fold ablation (§7c).

## How it was built

1. **Course-style search (§7):** start from the Class 2 tabular baseline, change one thing per experiment (Class 2), 3 seeds per configuration, blocked K-fold over years for the top configurations (`regression.ipynb`).
2. **Beyond the course templates (§7b):** weather-specific ideas evaluated with K-fold and paired *t*-statistics: 24-hour output, attention, multi-task min/max, Conv1D→GRU, a wider network. Classic ML benchmarks for comparison.
3. **Justification (§7c):** each component removed in turn and its cost measured.
4. **Final test:** retrain on 2006–2015, evaluate on 2016.

The workflow follows the Class 1 reproducibility checklist: settings in `config.yaml`, fixed seeds (123, 2137, 15190), normalisation fitted on training data only, results as mean ± std, and a reproducibility check. 327 training runs in total are logged in `runs/experiments.jsonl`.

**Transparency:** the test year was evaluated twice before, with earlier models that used a rain feature we later removed. Rain labels exist only in 2012 and 2016 (Issue 8 in the notebook). Those results are in `runs/history_with_rain_feature/`, and none of them was used for a modelling decision.

## Files

| File | What it is |
|---|---|
| `weather_forecast.ipynb` | **the whole project**: cleaning, exploration, features, model code, experiments, justification, final test and conclusions (follows the presentation outline in `Grading.md`) |
| `config.yaml` | every setting: the search plan, tailored ideas, classic benchmarks, justification ablation |
| `runs/` | experiment log, validation predictions of K-fold runs (`preds/`), results tables, final models and test predictions |
| `requirements.txt` | exact library versions |

## Reproduce

The dataset is not included in this repo. Download it from Kaggle ([arnab132/weather-dataset](https://www.kaggle.com/datasets/arnab132/weather-dataset), file `weatherHistory.csv`) into `data/`. `config.yaml` stores its SHA-256 hash, and the code checks it on load.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data && python -c "import kagglehub, shutil; shutil.copy(kagglehub.dataset_download('arnab132/weather-dataset') + '/weatherHistory.csv', 'data/')"
jupyter lab weather_forecast.ipynb # run all cells top to bottom (~5 min)
```

Finished training runs are reused from `runs/experiments.jsonl` and `runs/preds/`, with identical results, because every run is seeded and deterministic. Delete them to retrain everything from scratch (≈ 4.5 h of CPU training run one after another). TensorFlow doesn't support Python 3.14 yet, so use Python 3.12.

## Use of AI tools

Required by the course: the code and first drafts of the explanations were produced with Claude Code (Anthropic) and reviewed by the team. Details are at the end of the notebook.
