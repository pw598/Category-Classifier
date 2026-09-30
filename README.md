# Category Classifier

Predicts the four-level product category (Level 1–4 IDs) from product descriptions, with a
multinomial Naive Bayes benchmark and a sparse-input neural network over the same features.
Built to run on Databricks.

## Notebooks

| notebook | reads | writes |
|---|---|---|
| `01_pull_and_clean` | SQL pull, or a raw file in `data/` | `data/raw_data_<stamp>.txt`, `data/cleaned_data_<stamp>.txt`, `config/cleaning_<stamp>.json`, `config/cleaner_<stamp>.joblib`, `outputs/ngrams_<stamp>/` |
| `02_test_set_evaluation` | a cleaning run; a test-set ID file | `outputs/eval_<stamp>/`, `config/model_<stamp>.json` |
| `03_cross_validation` | a cleaning run | `outputs/cv_<stamp>/` (resumable), `config/model_<stamp>.json` |
| `04_train` | a model config and its cleaning run | `outputs/model_<stamp>/` (the deployable bundle) |
| `05_predict` | a model bundle; a file of IDs and descriptions, or a SQL pull | `outputs/predictions_<stamp>.txt` |

Every run gets a datestamp; nothing is overwritten. All parameters are set at the top of each
notebook. Each notebook prints the stamp or file name the next one needs.

## Layout

```
classifier/        the logic (no settings live here beyond defaults)
  cleaning.py      the cleaning recipe, pattern files, markers, n-gram counts
  features.py      rare-token pruning, count/tfidf vectorizers, order-insensitive bigrams
  hierarchy.py     label hierarchy; per-level probabilities -> valid paths
  nb.py            multinomial Naive Bayes held sparsely
  nn.py            sparse-input multi-head network (EmbeddingBag first layer), checkpointing
  model.py         model config, the classifier wrapper, bundle save/load
  crossval.py      k-fold / repeated holdout, one directory per run, resumable
  evaluation.py    accuracy overall, by level, split and confidence bucket
  plotting.py      charts
  io.py            files, JSON, ID lists, datestamps
cleaning_files/
  lists/           colours, sizes, units ... referenced by the pattern files
  patterns/        regex procedures; pick one with PATTERN_FILE in notebook 01
  abbrevs_to_expand.csv, to_normalize.csv, good_tokens.csv
data/ config/ outputs/
```

## Trying a different regex procedure

Copy a file in `cleaning_files/patterns/`, edit it, and set `PATTERN_FILE` to the copy in
notebook 01. `original.txt` is the NB repos' procedure; `nn_2026_09_28.txt` is the NN repo's
revision. The resolved patterns are stored in each cleaning config, so editing a file never
changes an earlier run.

## Requirements

`pip install -r requirements.txt`. `torch` is only needed for the neural network; `nltk`
installs itself on first use if missing. On Databricks, after pulling changes to
`classifier/`, restart Python (`dbutils.library.restartPython()`) so the new code is imported.
