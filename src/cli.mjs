#!/usr/bin/env node
// lexical-sophistication "message" [--strength S] [--seed N] [--max-tokens N] [--temperature T] [--top-k K]
//   [--top-p P] [--presence P]: Qwen3.5-0.8B's reply, its word choice steered toward common (S < 0) or rare (S > 0)
//   words, S from -3 to 3. The defaults are the live demo's.
import { readFile } from "node:fs/promises";
import { parseArgs } from "node:util";
import { createIndex, readTable, rarityPicker } from "./rarity.mjs";
import { Runner, tokenizer } from "./model.mjs";

// parseArgs reads "--strength -3" as two options: a negative number is joined to the option before it.
const args = process.argv.slice(2).reduce((out, a) => (/^-\d/.test(a) && out.at(-1)?.startsWith("--") ? [...out.slice(0, -1), `${out.at(-1)}=${a}`] : [...out, a]), []);
const { values, positionals } = parseArgs({
  args,
  allowPositionals: true,
  options: {
    strength: { type: "string", default: "0" },
    seed: { type: "string", default: "0" },
    "max-tokens": { type: "string", default: "256" },
    temperature: { type: "string", default: "1" },
    "top-k": { type: "string", default: "20" },
    "top-p": { type: "string", default: "1" },
    presence: { type: "string", default: "2" },
  },
});
if (positionals.length !== 1) {
  console.error('usage: lexical-sophistication "message" [--strength -3..3] [--seed 0] [--max-tokens 256]');
  process.exit(2);
}
const strength = Number(values.strength);
if (!(Math.abs(strength) <= 3)) throw new Error("strength must be from -3 to 3, as on the live demo");
const maxNewTokens = Number(values["max-tokens"]);
if (!(Number.isInteger(maxNewTokens) && maxNewTokens >= 1)) throw new Error("max tokens must be a positive integer");
const sampler = {
  temperature: Number(values.temperature), top_k: Number(values["top-k"]), top_p: Number(values["top-p"]),
  presence_penalty: Number(values.presence), seed: Number(values.seed),
};

const table = await readFile(new URL("../data/frequency-en.bin", import.meta.url));
const [index, t, runner] = await Promise.all([readTable(table).then(createIndex), tokenizer(), Runner.create()]);
const output = [];
let shown = "";
await runner.generate(t.prompt([{ role: "user", content: positionals[0] }]), {
  maxNewTokens, eos: t.eos, pick: rarityPicker(t, sampler, strength, index),
  onToken: (token) => {
    if (token === t.eos) return;
    output.push(token);
    const text = t.decode(output);
    if (text.endsWith("�")) return;  // a character split across tokens waits for its last one
    process.stdout.write(text.slice(shown.length));
    shown = text;
  },
});
process.stdout.write("\n");
