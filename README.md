# weather_forecast_DL

Final project for the Deep Learning course (Koźmiński University 2026/27, [`pamelon/DeepLearning-ALK`](https://github.com/pamelon/DeepLearning-ALK), `Grading.md`, option 3 – Weather Dataset).

**Task:** use the last 3 days of hourly weather data to predict tomorrow's temperature. `Apparent Temperature (C)` is not allowed.
**What we predict:** three numbers for tomorrow: the **mean**, the **lowest (min)** and the **highest (max)** temperature of the day.

## The project in one minute

1. **The bar to beat is "tomorrow = today".** Weather changes slowly, so repeating today's value is already a good forecast.
2. **We cleaned the data** and found 9 problems, including duplicate rows, sensor zeros, and fake copied days at the end of 2016.
3. **We compared 6 networks on all three targets** with the same fair test (K-fold over years), from the simple course network to one that reads the hourly and the daily data separately.
4. **The same kind of network wins all three targets:** two GRU branches. The mean is best predicted as a 24-hour curve; the min and max each get their own network.
5. **On unseen 2016 data, our models beat "tomorrow = today" on every target.**

## Result (test period 1 Jan – 3 Nov 2016, 308 days)

| Tomorrow's | Chosen network | Our model | Tomorrow = today | Better by |
|---|---|---|---|---|
| mean | two-branch GRU + attention, 24-hour curve | **1.15 °C** | 1.67 °C | 31 % |
| min | two-branch GRU, separate network | **1.13 °C** | 2.18 °C | 48 % |
| max | two-branch GRU + attention, separate network | **1.99 °C** | 2.55 °C | 22 % |

1. **Numbers:** average error in °C (MAE).
2. **K-fold** over 2006–2015 predicted the same improvements.
3. **Gradient-boosted trees** (the strongest classic method) are 5–13 % worse than our networks.
4. **Why the test ends on 3 November 2016:** after that date, every day in the dataset is a copy of the day before.

**The network:**
```
last 72 hours ──► GRU ──► attention (mean and max only) ──┐
last 3 days ────► GRU ────────────────────────────────────┼──► Dense(32) ──► Dropout(0.4) ──► change from today
tomorrow's day of year ───────────────────────────────────┘
```
Each final model is the average of 3 trainings with different seeds.

**Transparency:** earlier versions of the project also looked at the test period; the notebook's appendix lists every time.

## Files

| File | What it is |
|---|---|
| `weather_forecast.ipynb` | **the whole project**: data exploration, cleaning, features, models, comparison, final test and conclusions. It starts with a one-minute summary and a list of key terms, and follows a few example rows through every step of the data pipeline. |
| `config.yaml` | every setting: targets, shared settings, the networks to compare (with the reason for each), K-fold years |
| `runs/` | every training run, the chosen models, test predictions, and archives of earlier versions (`history_*`) |
| `requirements.txt` | exact library versions |
| `presentation/weather_forecast_presentation.pptx` | the 10-minute presentation (13 slides with speaker notes, plus a live look at the notebook) |

## Reproduce

1. The dataset is not included in this repo. Download it from Kaggle ([arnab132/weather-dataset](https://www.kaggle.com/datasets/arnab132/weather-dataset), file `weatherHistory.csv`) into `data/` (the third command below does this).
2. `config.yaml` stores the file's SHA-256 hash, and the code checks it on load.
3. Use Python 3.12: TensorFlow doesn't support 3.14 yet.
4. Run:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data && python -c "import kagglehub, shutil; shutil.copy(kagglehub.dataset_download('arnab132/weather-dataset') + '/weatherHistory.csv', 'data/')"
jupyter lab weather_forecast.ipynb # run all cells top to bottom (~35 min on a laptop)
```

5. Saved training runs in `runs/experiments.jsonl` and `runs/preds/` are reused, with identical results, because every run is repeatable.
6. Delete them to retrain everything from scratch (930 runs, about 5 hours on a laptop using 6 processes).

## Use of AI tools

The course (`Grading.md`) allows AI tools, but asks each team to say whether they used them and for what, and to be able to explain every part of the project.
1. **What we used:** Claude Code (Anthropic).
2. **What for:** writing the code and the first drafts of the explanations.
3. **How we checked it:** the team reviewed everything. Details are at the end of the notebook.
