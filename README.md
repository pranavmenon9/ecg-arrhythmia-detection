# ECG Arrhythmia Detection

Classify single ECG heartbeats into five types with a 1D CNN, tracked with MLflow, served through a FastAPI service, packaged with Docker, tested in CI and deployed on Render.

**Live demo:** https://ecg-arrhythmia-api.onrender.com/ (pick a sample beat or paste your own, and see the prediction)

**API docs:** https://ecg-arrhythmia-api.onrender.com/docs

> The service runs on a free tier and sleeps when idle, so the first request after a quiet period can take 30-60 seconds.
> This is a learning project, not a medical device.

## Results

Data: MIT-BIH heartbeat CSVs. Each row is one beat of 187 samples scaled to 0-1, with a label for one of five classes. The training file has 87,554 beats and is heavily imbalanced (83% Normal, under 1% Fusion). Models were compared on a stratified 20% validation split, and the chosen model was scored once on the separate test file (21,892 beats).

| Model | Class weights | Validation macro-F1 |
|---|---|---|
| MLP (187-128-64-5) | balanced | 0.712 |
| MLP (187-128-64-5) | sqrt of balanced | 0.879 |
| **1D CNN** | sqrt of balanced | **0.911** |

**Final test score of the 1D CNN: macro-F1 0.908, accuracy 98.1%.**

| Class | Precision | Recall | F1 | Test beats |
|---|---|---|---|---|
| Normal | 0.992 | 0.988 | 0.990 | 18,118 |
| Supraventricular | 0.774 | 0.826 | 0.799 | 556 |
| Ventricular | 0.954 | 0.950 | 0.952 | 1,448 |
| Fusion | 0.784 | 0.852 | 0.817 | 162 |
| Unknown | 0.977 | 0.989 | 0.983 | 1,608 |

What I learned along the way:
- Fully "balanced" class weights over-corrected: the model caught the rare classes but mislabelled many Normal beats. Taking the square root of the weights (a softer correction) lifted macro-F1 from 0.71 to 0.88.
- A 1D CNN, which looks at local shapes in the signal, beat the MLP on every class, most of all on the rare ones.
- Macro-F1 is used for early stopping and model comparison because accuracy hides poor performance on rare classes.

## Try the demo page

Open the live demo (or http://127.0.0.1:8000/ when running locally). Click one of the five sample beats, which are real beats from the MIT-BIH test set, to see the signal plotted and the model's probability for each class. You can also paste your own 187 values.

## Use the API

`POST /predict` takes exactly 187 numbers and returns the predicted class with a probability for each class. Invalid input (wrong length, non-numeric values) gets a `422` error.

```bash
curl -X POST https://ecg-arrhythmia-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d @app/sample_beat.json
```

```json
{"label": "Ventricular", "class_index": 2, "probabilities": {"Normal": 1.3e-09, "Supraventricular": 5.1e-08, "Ventricular": 1.0, "Fusion": 3.5e-11, "Unknown": 2.1e-13}}
```

`GET /health` returns `{"status": "ok", "model_loaded": true}`. Interactive docs are at `/docs`.

## Project structure

```
app/            FastAPI service (main.py), a sample beat, and the demo web page (static/)
src/            model.py (CNN class shared by training and the API), export_model.py
notebooks/      01 exploration, 03 MLP, 04 CNN, 05 final test evaluation
models/         cnn_sqrt.pt - the trained weights served by the API (879 KB)
tests/          API tests (pytest)
Dockerfile      container image for the API
.github/        CI workflow
data/           datasets (not tracked)
```

## MLOps pieces

- **Experiment tracking:** every notebook run logs parameters, per-epoch metrics, the classification report and the model to MLflow (SQLite backend).
- **Model registry:** `python -m src.export_model` saves the weights from the best run to `models/` and registers it in the MLflow Model Registry as `ecg-cnn` with the alias `champion`.
- **Tests:** 13 pytest tests cover the endpoints, output shape and probabilities, and input validation.
- **CI:** GitHub Actions runs the tests, builds the Docker image, starts the container and sends a real beat to `/predict` on every push.
- **Deployment:** Render builds the Dockerfile from this repo and serves it.

## Run it locally

Set up the environment (Python 3.11, CPU-only PyTorch):

```bash
conda create -n ecg python=3.11 -y
conda activate ecg
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn app.main:app
```

Then open http://127.0.0.1:8000/ for the demo page or http://127.0.0.1:8000/docs for the API docs.

Run the tests:

```bash
pytest
```

Build and run the container (needs Docker):

```bash
docker build -t ecg-api .
docker run -p 8000:8000 ecg-api
```

To retrain, download the MIT-BIH heartbeat CSVs (`mitbih_train.csv`, `mitbih_test.csv`) into `data/` and run the notebooks in order. Browse the logged runs with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Limitations

- The model was trained on pre-segmented, pre-scaled single beats from one dataset. It has not been tested on other recording devices, patients or raw ECG signals.
- Fusion and Supraventricular beats are the weakest classes, and the test set has only 162 Fusion beats, so those scores are noisy.
- The CSVs carry no patient IDs, so I could not split by patient. Beats from the same patient may appear in both the training and evaluation files, which can make the scores optimistic for a new patient.
