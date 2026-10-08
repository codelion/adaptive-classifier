# Deployment guide: multilingual models and non-Python runtimes

This guide covers two questions that come up often:

1. [Can I use it for languages other than English?](#multilingual-classification)
2. [Can I train in Python and run inference in C#, JavaScript or anything else with an ONNX runtime?](#running-outside-python)

Short answers: yes, and yes. You train and update in Python. Inference needs only the encoder in ONNX plus a few lines of arithmetic, which this guide spells out and `examples/portable_inference.py` implements.

---

## Multilingual classification

Nothing in the library is English-specific. The language support comes entirely from the encoder you pick, so swap the base model:

```python
from adaptive_classifier import AdaptiveClassifier

classifier = AdaptiveClassifier("intfloat/multilingual-e5-small")

classifier.add_examples(
    ["Das Produkt ist großartig", "Producto terrible, no lo recomiendo", "商品很好用"],
    ["positive", "negative", "positive"],
)
classifier.predict("Muy buen servicio")
```

Encoders worth trying (all published on the Hugging Face Hub):

| Model | Notes |
|---|---|
| `intfloat/multilingual-e5-small` / `-base` | Small and fast, about 100 languages. Mean pooling. |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Small, trained for sentence similarity. Mean pooling. |
| `BAAI/bge-m3` | Larger and stronger, long inputs, 100+ languages. CLS pooling. |
| `xlm-roberta-base` | Multilingual, but not trained for sentence similarity; expect weaker prototypes than the models above. |

Things that matter:

- **Pooling.** The default `pooling: "auto"` reads the model's own sentence-transformers config, so the table above is handled for you. The result is saved with the classifier. See the "Pooling" section of the README.
- **Prefixes.** Some models expect a prefix on every input (E5 wants `"query: "`). Add it yourself to training and prediction texts, consistently.
- **Mixed-language data works.** Good multilingual encoders place translations of the same sentence close together, so a class trained mostly on English examples will often recognise other languages. Add a few examples in each language you care about to firm it up.
- **Always evaluate on your own data.** Multilingual quality varies a lot by language and domain; the library does not ship per-language benchmarks.

---

## Running outside Python

### What gets saved

`classifier.save("./my_classifier")` (or `push_to_hub`) writes:

| File | Contents |
|---|---|
| `config.json` | Label maps (`id_to_label`), per-class `training_history`, and the `config` block: resolved `pooling`, `max_length`, blend weights |
| `tokenizer.json` and friends | The base model's tokenizer |
| `model.safetensors` | `prototype_<label>` tensors (one vector per class) and `adaptive_head_model.{0,3,6}.{weight,bias}` for the neural head |
| `examples.json` | A few representative training examples per class. Not needed for inference |
| `onnx/model.onnx`, `onnx/model_quantized.onnx` | The transformer encoder only. It returns token embeddings, not a sentence vector or class scores |

The encoder is the only large file. Everything else is tiny and framework-neutral (JSON plus safetensors, which has readers in most languages).

`save()` writes the tokenizer files (`tokenizer.json`, `tokenizer_config.json`, ...) next to the model, and records the **resolved** pooling mode (`mean` or `cls`) in `config.json`, so the directory is self-contained. Nothing needs to be looked up from the base model at inference time.

Directories saved by 0.2.0 or earlier are different. They have no tokenizer files, and their `config.json` has either no `pooling` key (meaning `cls`) or the literal `"auto"`. Re-save them with the current version, or generate the tokenizer once with `AutoTokenizer.from_pretrained("<base model>").save_pretrained("./my_classifier")` and pass the pooling mode to the reference script yourself (`mean` for most sentence-transformers models, `cls` for BGE).

### The inference algorithm

Given the files above, `predict(text)` is:

1. **Tokenize** with the base model's tokenizer, truncating to `config.max_length` (default 512).
2. **Run the ONNX encoder** to get `last_hidden_state`, shape `(1, tokens, dim)`. Feed `input_ids` and `attention_mask`, plus `token_type_ids` if the graph declares that input (BERT-family models do).
3. **Pool** to one vector:
   - `mean`: sum of token vectors weighted by `attention_mask`, divided by the number of real tokens.
   - `cls`: the first token's vector.
4. **L2-normalise** the vector.
5. **Prototype scores.** For each class prototype `p`, compute `exp(-||p - v||²)` (squared Euclidean distance), then take a softmax across classes.
6. **Neural head scores** (skip if `model.safetensors` has no head). Compute `h1 = ReLU(W0·v + b0)`, `h2 = ReLU(W3·h1 + b3)`, `logits = W6·h2 + b6`, then softmax. Dropout is inactive at inference. Weights are stored as `(out, in)`.
7. **Blend per class.** `score = proto_weight · proto_score + neural_weight · head_score`. A class with fewer than `new_class_example_threshold` (default 10) entries in `training_history` uses the `new_class_*` weights (default 0.3 / 0.7) instead of `prototype_weight` / `neural_weight` (default 0.7 / 0.3).
8. **Normalise** the blended scores to sum to 1 and sort descending.

This applies to the default (non-strategic) mode. Strategic mode adds extra steps that are not covered here.

`examples/portable_inference.py` implements exactly these steps with NumPy, `onnxruntime` and `tokenizers`, with no PyTorch or FAISS:

```bash
pip install onnxruntime numpy safetensors tokenizers
python examples/portable_inference.py ./my_classifier "text to classify"
```

It is checked against `AdaptiveClassifier.predict` on the same saved directory with both `mean` and `cls` pooling (`tests/test_portable_export.py`), and agrees to float32 rounding. Use it as the ground truth when you port: run the same inputs through both and compare scores.

### Quantized or full-precision encoder

`onnx/model_quantized.onnx` is INT8 and about 4x smaller. Its scores differ slightly from the full-precision graph, so check that your class ranking survives on your own data. `onnx/model.onnx` is the full-precision graph.

### .NET (C#)

Load the encoder with the `Microsoft.ML.OnnxRuntime` NuGet package, and tokenize with `Microsoft.ML.Tokenizers` (or any library that can read your model's `vocab.txt` or `tokenizer.json`).

```csharp
using var session = new InferenceSession("my_classifier/onnx/model_quantized.onnx");
var inputs = new List<NamedOnnxValue> {
    NamedOnnxValue.CreateFromTensor("input_ids",      new DenseTensor<long>(ids,  new[] { 1, ids.Length })),
    NamedOnnxValue.CreateFromTensor("attention_mask", new DenseTensor<long>(mask, new[] { 1, ids.Length })),
    // add "token_type_ids" (all zeros for single sentences) if session.InputMetadata contains it
};
using var results = session.Run(inputs);
var hidden = results.First().AsTensor<float>();   // (1, tokens, dim)
// then pool, normalise, and score as in steps 3-8 above
```

The scoring is a few dozen lines of array maths. Read `model.safetensors` with a safetensors reader (the format is an 8-byte header length, a JSON header, then raw little-endian float32 data), or export the prototypes and head weights to JSON once in Python and ship that.

### JavaScript / Node / browser

Use `onnxruntime-node` (server) or `onnxruntime-web` (browser) for the encoder and `@huggingface/transformers` for tokenization.

```js
import * as ort from "onnxruntime-node";
const session = await ort.InferenceSession.create("my_classifier/onnx/model_quantized.onnx");
const out = await session.run({
  input_ids:      new ort.Tensor("int64", BigInt64Array.from(ids),  [1, ids.length]),
  attention_mask: new ort.Tensor("int64", BigInt64Array.from(mask), [1, ids.length]),
  // token_type_ids too if session.inputNames includes it
});
const hidden = out.last_hidden_state;   // Float32Array data, dims [1, tokens, dim]
// then pool, normalise, and score as in steps 3-8 above
```

For the browser, check the size of `model_quantized.onnx` first: multilingual encoders carry a large vocabulary embedding and can be well over 100 MB even when quantized.

The C# and JavaScript snippets above are sketches of the API calls and have **not** been run; the Python reference script is the tested artifact.

### Updating a deployed model

Inference outside Python is read-only. The workflow that works well:

1. Keep the Python classifier as the source of truth. Add examples and classes there (`add_examples`) as new data arrives.
2. `save()` again and ship the updated `config.json` and `model.safetensors`. If the encoder is unchanged, `onnx/` does not need to be redistributed.
3. Reload them in the non-Python service.

### Known limitations

- Strategic-mode predictions and multi-label classifiers (`MultiLabelAdaptiveClassifier`) have different scoring and are not covered by the reference implementation.
- The encoder must be one that exports to ONNX with a `last_hidden_state` output, which is the case for standard BERT-family encoders. Custom architectures that need `trust_remote_code` may not export cleanly.
