// Qwen3.5-0.8B on the CPU with ONNX Runtime: onnx-community's 4-bit graphs, the files of the live demo, read from the
// Hugging Face cache when they are there, else downloaded once into ~/.cache/lexical-sophistication.
import { existsSync, mkdirSync, createWriteStream, renameSync, statSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join, dirname } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import * as ort from "onnxruntime-node";
import { loadTokenizer } from "./page/chat.mjs";

const MODEL = { repo: "Qwen/Qwen3.5-0.8B", revision: "2fc06364715b967f1860aea9cf38778875588b17" };
const ONNX = { repo: "onnx-community/Qwen3.5-0.8B-ONNX-OPT", revision: "fafab72d87a9e6be3925b38caf48286d2838f2d0" };
const HUB = process.env.HF_HUB_CACHE ?? join(process.env.HF_HOME ?? join(homedir(), ".cache/huggingface"), "hub");
const OWN = join(homedir(), ".cache/lexical-sophistication");

// onnxruntime-node 1.31 reads a Float16Array (Node 24 has them) as empty. Hidden while ONNX Runtime picks its float16
// array type at the first tensor, so it keeps float16 data as their bits in a Uint16Array. This holds only if this
// module is imported before anything else creates an ONNX Runtime tensor.
const float16 = globalThis.Float16Array;
delete globalThis.Float16Array;
new ort.Tensor("float32", new Float32Array(1), [1]);
globalThis.Float16Array = float16;
if (!(new ort.Tensor("float16", new Uint16Array(1), [1]).data instanceof Uint16Array)) throw new Error("import model.mjs before creating any ONNX Runtime tensor");
const floats = (tensor) => (tensor.type === "float16"
  ? Float32Array.from(new Float16Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length))
  : Float32Array.from(tensor.data));

// Files of a pinned repository, all from the same folder: the Hugging Face cache's snapshot if it holds every one, else
// OWN, which downloads those it lacks. An ONNX graph reads its weights from beside it, so the two never split.
async function files({ repo, revision }, names) {
  const snapshot = join(HUB, `models--${repo.replace("/", "--")}`, "snapshots", revision);
  if (names.every((name) => existsSync(join(snapshot, name)))) return names.map((name) => join(snapshot, name));
  return Promise.all(names.map(async (name) => {
    const path = join(OWN, repo, revision, name);
    if (existsSync(path)) return path;
    mkdirSync(dirname(path), { recursive: true });
    const response = await fetch(`https://huggingface.co/${repo}/resolve/${revision}/${name}`);
    if (!response.ok) throw new Error(`${repo}/${name}: HTTP ${response.status}`);
    process.stderr.write(`downloading ${name}\n`);
    await pipeline(Readable.fromWeb(response.body), createWriteStream(path + ".part"));
    const expected = Number(response.headers.get("content-length"));
    if (expected && statSync(path + ".part").size !== expected) throw new Error(`${repo}/${name}: the download was cut short; run again`);
    renameSync(path + ".part", path);
    return path;
  }));
}

export async function tokenizer() {
  return loadTokenizer(async (name) => readFile((await files(MODEL, [name]))[0], "utf8"));
}

export class Runner {
  static async create(variant = "q4f16") {
    const session = async (name) => {
      const [graph] = await files(ONNX, [`onnx/${name}_${variant}.onnx`, `onnx/${name}_${variant}.onnx_data`]);
      return ort.InferenceSession.create(graph);
    };
    return new Runner(...(await Promise.all([session("embed_tokens"), session("decoder_model_merged")])));
  }

  constructor(embed, decoder) {
    this.embed = embed;
    this.decoder = decoder;
    // Empty state from the graph's own input shapes: no past tokens, batch 1.
    this.past = decoder.inputMetadata.filter((m) => m.name.startsWith("past_")).map((m) => ({
      name: m.name, type: m.type,
      dims: m.shape.map((d) => (d === "past_sequence_length" ? 0 : typeof d === "string" ? 1 : d)),
    }));
    this.keep = new ort.Tensor("int64", BigInt64Array.from([1n]), []);  // num_logits_to_keep
  }

  fresh() {
    const past = Object.fromEntries(this.past.map(({ name, type, dims }) => {
      const size = dims.reduce((a, b) => a * b, 1);
      return [name, new ort.Tensor(type, type === "float16" ? new Uint16Array(size) : new Float32Array(size), dims)];
    }));
    return { past, length: 0 };
  }

  // Decodes ids after state: {logits: Float32Array [vocab], state}.
  async step(ids, state) {
    const n = ids.length, past = state.length;
    const { inputs_embeds } = await this.embed.run({ input_ids: new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, n]) });
    const out = await this.decoder.run({
      inputs_embeds,
      attention_mask: new ort.Tensor("int64", new BigInt64Array(past + n).fill(1n), [1, past + n]),
      position_ids: new ort.Tensor("int64", BigInt64Array.from({ length: 3 * n }, (_, i) => BigInt(past + (i % n))), [3, 1, n]),
      num_logits_to_keep: this.keep,
      ...state.past,
    });
    const next = {};
    for (const [name, value] of Object.entries(out)) {
      if (name.startsWith("present")) next[name.replace("present_", "past_").replace("present.", "past_key_values.")] = value;
    }
    return { logits: floats(out.logits), state: { past: next, length: past + n } };
  }

  // One reply: token by token until eos or maxNewTokens; pick(logits) chooses each token, and onToken(token) gets
  // each one but eos.
  async generate(ids, { maxNewTokens, eos, pick, onToken = () => {} }) {
    let { logits, state } = await this.step(ids, this.fresh());
    const out = [];
    for (;;) {
      const token = pick(logits);
      out.push(token);
      if (token === eos) return out;
      onToken(token);
      if (out.length >= maxNewTokens) return out;
      ({ logits, state } = await this.step([token], state));
    }
  }
}
