# Few-shot benchmark: adaptive-classifier vs SetFit vs logistic regression

`scripts/benchmark_fewshot.py` measures how well each method does when it only has a handful
of labelled examples per class, which is the situation this library is built for. It runs as a
manually triggered GitHub Actions workflow, or on your own machine.

## What is compared

All methods use the **same sentence encoder** (default `sentence-transformers/all-MiniLM-L6-v2`),
the **same stratified training sample**, and the **same test examples**.

| method | what it is |
|---|---|
| `adaptive` | this library with default settings |
| `adaptive-proto` | this library with the neural head switched off (prototypes only) |
| `logreg` | logistic regression on the frozen, L2-normalised embeddings |
| `setfit` | [SetFit](https://github.com/huggingface/setfit): contrastive fine-tuning of the encoder, then a logistic-regression head |

`adaptive-proto` and `logreg` are included so that a difference between `adaptive` and `setfit`
can be traced: is it the encoder fine-tuning, or the classifier on top?

## Metrics

| metric | meaning |
|---|---|
| accuracy, macro-F1 | on a fixed random subset of the test split (`--max-test`, default 1000) |
| ECE | expected calibration error of the reported confidences: do 80% confidences come true 80% of the time? Lower is better. Reported **without** any calibration step |
| train (s) | time to train from raw text, including embedding the training examples (model loading is not counted) |
| predict test set (s), latency p50 (ms) | time for the whole test set from raw text, and the median time for a single query |
| add one example (s) | the cost of learning one more labelled example. `adaptive` updates in place. `logreg` refits on cached features. **SetFit has no incremental update, so its full training time is reported** and marked "(full retrain)" |

Each (dataset, shots, seed) draws a stratified sample; a smaller sample for a seed is always a
subset of a larger one, so results across shot counts are comparable. Results are mean ± standard
deviation over seeds. The best value in each column is bold.

## Running it on GitHub

Actions → **Few-shot benchmark** → **Run workflow**. The inputs are validated before any long job
starts, and each dataset is checked to load, so a typo fails in seconds. Each dataset runs as its
own job (in parallel); the combined Markdown report appears in the run summary and as the
`benchmark-report` artifact, and the raw numbers as `results-<dataset>` artifacts.

Defaults (`ag_news,emotion`, 4 and 16 shots, 3 seeds) take a while on GitHub's CPU runners, mostly
because SetFit fine-tunes the encoder. `banking77` (77 classes) is much slower: use few shots and
one seed first.

## Running it locally

```bash
pip install -e ".[benchmark]"
python scripts/benchmark_fewshot.py check ag_news emotion banking77        # do the datasets load?
python scripts/benchmark_fewshot.py run --dataset ag_news --shots 4 16 --seeds 3 --out results/ag_news.json
python scripts/benchmark_fewshot.py report results/*.json --out results/summary.md
```

`--methods` selects a subset (for example `adaptive logreg` to skip SetFit), `--model` changes the
encoder, and `--dataset synthetic` runs a tiny generated dataset with no network access, only to
check that the harness works.

## Datasets

| name | source | classes |
|---|---|---|
| `ag_news` | `fancyzhx/ag_news` | 4 |
| `emotion` | `dair-ai/emotion` | 6 |
| `banking77` | `mteb/banking77` | 77 |

`PolyAI/banking77` is not used: it is a Hub loading script, which current `datasets` releases
refuse to run. The `mteb` copy has the same data as parquet. To add a dataset, add an entry to
`DATASETS` in the script (Hub id, text column, label column, test split name) and run `check`.

## Limits of what this shows

Read the numbers with these in mind:

- **One encoder.** Everything shares MiniLM-L6. A different encoder can change the ranking, and
  SetFit is usually run with larger ones.
- **Untuned settings.** SetFit uses its common defaults (1 epoch, 20 pair iterations, batch 16);
  logistic regression uses `C=10`; adaptive-classifier uses its defaults. None was tuned per dataset.
- **Few seeds, subsampled test set.** Differences of a point or two are within noise. With 3 seeds
  the standard deviations are themselves rough.
- **Three classification datasets of short English text.** Nothing here speaks to long documents,
  other languages, or very many classes beyond `banking77`.
- **CPU timings are indicative.** They depend on the machine and on whether a GPU is present
  (the report records the device).
- **No LLM baseline.** Few-shot prompting of a language model is a real alternative and is not
  included.
- **The first full runs have not been analysed yet.** Check the report before quoting any number
  from it, and look at the run's `error` fields for failed cells (shown as `failed`).

## Checking the harness itself

The parts that can be tested offline (sampling, metrics, aggregation, the report, the command line,
input validation, and a smoke run of every method on a tiny local model) are covered by
`tests/test_benchmark_harness.py`, so a surprising number is more likely to be about the methods than
about the harness.
