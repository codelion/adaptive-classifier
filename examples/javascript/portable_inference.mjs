// Run a saved AdaptiveClassifier from Node.js, with no Python, PyTorch or FAISS.
//
//   npm install
//   node portable_inference.mjs ./my_classifier "text to classify"
//
// This is a port of examples/portable_inference.py; docs/deployment.md explains each
// step. It reads config.json, model.safetensors, tokenizer.json, tokenizer_config.json
// and onnx/ from a directory written by AdaptiveClassifier.save().

import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { Tokenizer } from "@huggingface/tokenizers";

// onnxruntime-node is the fast server runtime; onnxruntime-web (WASM) also runs in
// Node and is the fallback where the native binaries are unavailable.
async function loadRuntime() {
  for (const name of ["onnxruntime-node", "onnxruntime-web"]) {
    try {
      return await import(name);
    } catch {
      // try the next one
    }
  }
  throw new Error("Install onnxruntime-node or onnxruntime-web");
}

// safetensors: 8-byte little-endian header length, a JSON header, then raw data.
export function readSafetensors(path) {
  const buffer = readFileSync(path);
  const headerLength = Number(buffer.readBigUInt64LE(0));
  const header = JSON.parse(buffer.subarray(8, 8 + headerLength).toString("utf8"));
  const base = 8 + headerLength;
  const tensors = {};
  for (const [name, info] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    if (info.dtype !== "F32") throw new Error(`${name}: unsupported dtype ${info.dtype}`);
    const [start, end] = info.data_offsets;
    // Copy: the file buffer is not guaranteed to be 4-byte aligned.
    const bytes = buffer.subarray(base + start, base + end);
    const data = new Float32Array(bytes.byteLength / 4);
    new Uint8Array(data.buffer).set(bytes);
    tensors[name] = { shape: info.shape, data };
  }
  return tensors;
}

function softmax(values) {
  const max = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - max));
  const total = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / total);
}

function squaredDistance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += (a[i] - b[i]) ** 2;
  return sum;
}

// out = W x + b with W stored (out, in).
function affine({ shape, data }, bias, x) {
  const [rows, cols] = shape;
  const out = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    let sum = bias.data[r];
    for (let c = 0; c < cols; c++) sum += data[r * cols + c] * x[c];
    out[r] = sum;
  }
  return out;
}

export class PortableClassifier {
  static async load(directory, { pooling, preferQuantized = true } = {}) {
    const self = new PortableClassifier();
    const cfg = JSON.parse(readFileSync(join(directory, "config.json"), "utf8"));
    const settings = cfg.config ?? {};
    self.cfg = cfg;
    self.maxLength = settings.max_length ?? 512;
    self.prototypeWeight = settings.prototype_weight ?? 0.7;
    self.neuralWeight = settings.neural_weight ?? 0.3;
    self.newThreshold = settings.new_class_example_threshold ?? 10;
    self.newPrototypeWeight = settings.new_class_prototype_weight ?? null;
    self.newNeuralWeight = settings.new_class_neural_weight ?? null;
    self.temperature = settings.prototype_temperature ?? null;
    self.pooling = pooling ?? settings.pooling;
    if (self.pooling !== "mean" && self.pooling !== "cls") {
      throw new Error("config.json does not fix a pooling mode; pass { pooling: 'mean' | 'cls' }");
    }

    const count = Object.keys(cfg.id_to_label).length;
    self.labels = Array.from({ length: count }, (_, i) => cfg.id_to_label[String(i)]);
    const tensors = readSafetensors(join(directory, "model.safetensors"));
    self.prototypes = new Map();
    for (const label of self.labels) {
      const tensor = tensors[`prototype_${label}`];
      if (tensor) self.prototypes.set(label, tensor.data);
    }
    // Linear-ReLU-Dropout x2 then Linear: affine layers at indices 0, 3 and 6.
    self.head = tensors["adaptive_head_model.0.weight"]
      ? [0, 3, 6].map((i) => [tensors[`adaptive_head_model.${i}.weight`], tensors[`adaptive_head_model.${i}.bias`]])
      : null;

    self.tokenizer = new Tokenizer(
      JSON.parse(readFileSync(join(directory, "tokenizer.json"), "utf8")),
      JSON.parse(readFileSync(join(directory, "tokenizer_config.json"), "utf8")),
    );

    self.ort = await loadRuntime();
    const quantized = join(directory, "onnx", "model_quantized.onnx");
    const full = join(directory, "onnx", "model.onnx");
    const file = preferQuantized && existsSync(quantized) ? quantized : full;
    self.session = await self.ort.InferenceSession.create(file);
    return self;
  }

  async embed(text) {
    const encoded = this.tokenizer.encode(text);
    let ids = encoded.ids;
    let mask = encoded.attention_mask;
    if (ids.length > this.maxLength) {
      // Keep the final special token (e.g. [SEP]) as the Python tokenizer does.
      ids = [...ids.slice(0, this.maxLength - 1), ids[ids.length - 1]];
      mask = mask.slice(0, this.maxLength);
    }
    const n = ids.length;
    const tensor = (values) => new this.ort.Tensor("int64", BigInt64Array.from(values.map(BigInt)), [1, n]);
    const feeds = { input_ids: tensor(ids), attention_mask: tensor(mask) };
    if (this.session.inputNames.includes("token_type_ids")) {
      feeds.token_type_ids = tensor(new Array(n).fill(0));
    }
    const output = await this.session.run(feeds);
    const hidden = output[this.session.outputNames[0]]; // (1, tokens, dim)
    const dim = hidden.dims[2];

    const vec = new Float64Array(dim);
    if (this.pooling === "cls") {
      for (let d = 0; d < dim; d++) vec[d] = hidden.data[d];
    } else {
      let real = 0;
      for (let t = 0; t < n; t++) {
        if (!mask[t]) continue;
        real++;
        for (let d = 0; d < dim; d++) vec[d] += hidden.data[t * dim + d];
      }
      for (let d = 0; d < dim; d++) vec[d] /= Math.max(real, 1e-9);
    }
    const norm = Math.max(Math.sqrt(vec.reduce((s, v) => s + v * v, 0)), 1e-12);
    return Float32Array.from(vec, (v) => v / norm);
  }

  headProbabilities(embedding) {
    let x = embedding;
    this.head.forEach(([weight, bias], i) => {
      x = affine(weight, bias, x);
      if (i < this.head.length - 1) x = x.map((v) => Math.max(v, 0));
    });
    return softmax(Array.from(x));
  }

  weights(label) {
    const trained = this.cfg.training_history?.[label] ?? 0;
    if (trained >= this.newThreshold) return [this.prototypeWeight, this.neuralWeight];
    if (this.newPrototypeWeight !== null && this.newNeuralWeight !== null) {
      return [this.newPrototypeWeight, this.newNeuralWeight];
    }
    const share = trained / this.newThreshold;
    return [1 - share + share * this.prototypeWeight, share * this.neuralWeight];
  }

  async predict(text, k = 5) {
    const embedding = await this.embed(text);

    const names = [...this.prototypes.keys()];
    const distances = names.map((name) => squaredDistance(this.prototypes.get(name), embedding));
    const prototypeScores = new Map(
      names.map((name, i) => [
        name,
        (this.temperature
          ? softmax(distances.map((d) => -d / this.temperature))
          : softmax(distances.map((d) => Math.exp(-d))))[i],
      ]),
    );
    const headScores = this.head ? this.headProbabilities(embedding) : null;

    const combined = this.labels.map((label, i) => {
      const [wp, wn] = this.weights(label);
      return [label, (prototypeScores.get(label) ?? 0) * wp + (headScores ? headScores[i] : 0) * wn];
    });
    const total = combined.reduce((s, [, score]) => s + score, 0);
    return combined
      .sort((a, b) => b[1] - a[1])
      .slice(0, k)
      .map(([label, score]) => [label, score / total]);
  }
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  if (process.argv.length < 4) {
    console.error("usage: node portable_inference.mjs <classifier dir> <text>");
    process.exit(2);
  }
  const classifier = await PortableClassifier.load(process.argv[2]);
  for (const [label, score] of await classifier.predict(process.argv.slice(3).join(" "))) {
    console.log(`${label}\t${score.toFixed(4)}`);
  }
}
