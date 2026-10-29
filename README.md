# weather_forecast_DL

Final project for the Deep Learning course (Koźmiński University 2026/27, [`pamelon/DeepLearning-ALK`](https://github.com/pamelon/DeepLearning-ALK), `Grading.md`, option 3 – Weather Dataset).

**Goal:** from the last 3 days of weather, predict the next day's temperature (regression). `Apparent Temperature (C)` is never used.
**Our targets:** the next day's **mean, minimum and maximum** temperature (local calendar day). Each has its own naive baseline, "tomorrow = today".

## Result (unseen test period 1 Jan – 3 Nov 2016, 308 days)

| Next-day | Chosen network | Model (3-seed average) | Tomorrow = today | Improvement | Climatology |
|---|---|---|---|---|---|
| mean | two-branch GRU + attention, predicting the 24-hour curve | **1.152** | 1.673 | −31 % | 2.842 |
| min | two-branch GRU, separate model | **1.126** | 2.185 | −48 % | 2.769 |
| max | two-branch GRU + attention, separate model | **1.988** | 2.553 | −22 % | 3.679 |

MAE in °C. The K-fold estimates over 2006–2015 were 1.123 / 1.167 / 1.824, with the same relative gains. On the same folds, gradient-boosted trees reach 1.221 / 1.371 / 1.946.

**Different targets, different winners.** All three are two-branch GRU networks. The best output design differs by target:
- **Mean:** best taken from a predicted 24-hour curve.
- **Min and max:** need their own models, because a curve trained with MSE flattens its peaks and lows.

**Why the test period ends on 3 November 2016:** from 4 November to 31 December 2016, every day in the dataset is an almost exact copy of the previous day. These artificial days were found during the analysis (Issue 9) and are excluded from evaluation. No other year has a single copied day.

**Network:**
- Inputs: a GRU over the 72 hourly readings (optionally with attention pooling), a GRU over 3 daily summary rows, and the day of the year.
- Head: `Dense(32, relu)` → `Dropout(0.4)` → output.
- The output is the change from today's value: 24 hourly values for the mean model, 1 value for the min and max models.
- Each final model is the average of 3 seeds.

## How it was built (notebook sections)

1. **EDA and cleaning (§2–§3):** 9 data issues, with a keep/transform/drop decision for every column.
2. **Baselines (§6):** "tomorrow = today" is the bar to beat for each target.
3. **One comparison (§7):**
   - 6 networks: the Class 2 Dense baseline, Dense with the shared settings, daily GRU, hourly GRU, two-branch GRU, and two-branch GRU + attention.
   - 3 output designs: a separate model per target, one joint 3-output model, a 24-hour curve.
   - All 3 targets, plus classic ML, evaluated with blocked K-fold over years (5 folds × 3 seeds) and paired *t*-statistics. The best network is chosen per target.
4. **Justification (§8):** each component and each feature group removed in turn, with its cost measured.
5. **Final test (§10):** each chosen network retrained on 2006–2015, evaluated on 1 Jan – 3 Nov 2016.

The workflow follows the Class 1 reproducibility checklist: settings in `config.yaml`, fixed seeds (123, 2137, 15190), normalisation fitted on training data only, results as mean ± std, and a reproducibility check. All 930 K-fold training runs are logged in `runs/experiments.jsonl`.

**Transparency:**
- The test period was evaluated by earlier versions of the project. The notebook's appendix lists every evaluation, and their results are archived in `runs/history_*`.
- The model for the mean is identical to the previous version's final model, so its test score was already known. The min and max models were evaluated for the first time.

## Files

| File | What it is |
|---|---|
| `weather_forecast.ipynb` | **the whole project**: exploratory data analysis with keep/drop decisions, cleaning, features, model code, experiments, justification, final test and conclusions (follows the presentation outline in `Grading.md`) |
| `config.yaml` | every setting: targets, shared settings, candidate networks (with the reason for each), output designs, K-fold folds, classic benchmarks, justification ablation |
| `runs/` | experiment log, validation predictions of K-fold runs (`preds/`), chosen configurations, justification table, final models and test predictions (`final/`), archives of earlier versions (`history_*`) |
| `requirements.txt` | exact library versions |

## Reproduce

The dataset is not included in this repo. Download it from Kaggle ([arnab132/weather-dataset](https://www.kaggle.com/datasets/arnab132/weather-dataset), file `weatherHistory.csv`) into `data/`. `config.yaml` stores its SHA-256 hash, and the code checks it on load.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data && python -c "import kagglehub, shutil; shutil.copy(kagglehub.dataset_download('arnab132/weather-dataset') + '/weatherHistory.csv', 'data/')"
jupyter lab weather_forecast.ipynb # run all cells top to bottom (~35 min on CPU, mostly the 9 final models)
```

Finished training runs are reused from `runs/experiments.jsonl` and `runs/preds/`, with identical results, because every run is seeded and deterministic. Delete them to retrain everything from scratch: 930 K-fold runs, about 5 h on a laptop CPU with 6 parallel processes, much longer one after another. TensorFlow doesn't support Python 3.14 yet, so use Python 3.12.

## Use of AI tools

Required by the course: the code and first drafts of the explanations were produced with Claude Code (Anthropic) and reviewed by the team. Details are at the end of the notebook.
