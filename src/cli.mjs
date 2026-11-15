#!/usr/bin/env node
// lexical-sophistication "message" [--strength S] [--seed N] [--max-tokens N] [--temperature T] [--top-k K]
//   [--top-p P] [--presence P]: Qwen3.5-0.8B's reply, its word choice steered toward common (S < 0) or rare (S > 0)
//   words, S from -3 to 3. The defaults are the live demo's.
import { readFile } from "node:fs/promises";
import { parseArgs } from "node:util";
import { createIndex, readTable, rarityPicker } from "./rarity.mjs";
import { validateSampler } from "./page/sampling.mjs";
import { Runner, tokenizer } from "./model.mjs";

const OPTIONS = { strength: "0", seed: "0", "max-tokens": "256", temperature: "1", "top-k": "20", "top-p": "1", presence: "2" };
const USAGE = 'usage: lexical-sophistication "message" [--strength -3..3] [--seed 0] [--max-tokens 256]';
const fail = (message) => {
  console.error(message);
  process.exit(2);
};

// parseArgs reads "--strength -3" as two options, so a negative number is joined to the value option before it.
const args = [];
for (const a of process.argv.slice(2)) {
  const option = args.at(-1)?.match(/^--([\w-]+)$/)?.[1];
  if (option in OPTIONS && /^-\.?\d/.test(a)) args[args.length - 1] += `=${a}`;
  else args.push(a);
}
let parsed;
try {
  parsed = parseArgs({ args, allowPositionals: true, options: Object.fromEntries(Object.entries(OPTIONS).map(([k, v]) => [k, { type: "string", default: v }])) });
} catch (error) {
  fail(`${error.message}\n${USAGE}`);
}
const { values, positionals } = parsed;
if (positionals.length !== 1) fail(USAGE);
const strength = Number(values.strength);
if (!(Math.abs(strength) <= 3)) fail("strength must be from -3 to 3, as on the live demo");
const maxNewTokens = Number(values["max-tokens"]);
if (!(Number.isInteger(maxNewTokens) && maxNewTokens >= 1)) fail("max tokens must be a positive integer");
const sampler = {
  temperature: Number(values.temperature), top_k: Number(values["top-k"]), top_p: Number(values["top-p"]),
  presence_penalty: Number(values.presence), seed: Number(values.seed),
};
try {
  validateSampler(sampler);  // before the model loads
} catch (error) {
  fail(error.message);
}

const table = await readFile(new URL("../data/frequency-en.bin", import.meta.url));
const [index, t, runner] = await Promise.all([readTable(table).then(createIndex), tokenizer(), Runner.create()]);
// Printed as it comes: the tokens since the last print are decoded together, and held while they end inside a character.
let pending = [];
const print = () => {
  const text = t.decode(pending);
  if (text.endsWith("�")) return;
  process.stdout.write(text);
  pending = [];
};
await runner.generate(t.prompt([{ role: "user", content: positionals[0] }]), {
  maxNewTokens, eos: t.eos, pick: rarityPicker(t, sampler, strength, index),
  onToken: (token) => {
    pending.push(token);
    print();
  },
});
process.stdout.write((pending.length ? t.decode(pending) : "") + "\n");  // the tokenizer refuses to decode nothing
