// Pins this repository to the live demo. next-token.json was measured on the page (Edge, WebGPU, q4f16): after the
// chat prompt and the start of a reply, the model's 20 most likely next tokens and their probabilities once the
// sampler's bonus is applied, at strengths -3, -1, 0, 1 and 3.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createIndex, readTable, candidates, probabilities, rarityPicker } from "../src/rarity.mjs";
import { picker } from "../src/page/sampling.mjs";
import { Runner, tokenizer } from "../src/model.mjs";

const { cases, results } = JSON.parse(readFileSync(new URL("next-token.json", import.meta.url), "utf8"));
const index = createIndex(await readTable(readFileSync(new URL("../data/frequency-en.bin", import.meta.url))));

test("the sampler and table give the page's probabilities from the page's logits", () => {
  for (const r of results) {
    const tokens = candidates(index, r.prefix, r.rows);
    for (const s of cases.strengths) {
      probabilities(tokens, s).forEach((p, i) => assert.ok(Math.abs(p - r.settings[s][i].p) < 1e-4, `${r.prefix} ${s} ${r.rows[i].piece}`));
    }
  }
});

test("the model on the CPU gives the page's next tokens", async () => {
  const t = await tokenizer(), runner = await Runner.create();
  for (const r of results) {
    const ids = [...t.prompt([{ role: "user", content: cases.message }]), ...t.encode(r.prefix)];
    const { logits } = await runner.step(ids, runner.fresh());
    const top = [...logits.keys()].sort((a, b) => logits[b] - logits[a]).slice(0, r.rows.length);
    assert.deepEqual(new Set(top), new Set(r.rows.map((x) => x.id)), r.prefix);
    for (const x of r.rows) assert.ok(Math.abs(logits[x.id] - x.logit) < 0.15, `${r.prefix} ${x.piece}: ${logits[x.id]} vs ${x.logit}`);
  }
  // Strength 0 is the plain sampler, draw for draw.
  const sampler = { temperature: 1, top_k: 20, top_p: 1, presence_penalty: 2, seed: 0 }, ids = t.prompt([{ role: "user", content: cases.message }]);
  const plain = await runner.generate(ids, { maxNewTokens: 24, eos: t.eos, sampler });
  const zero = await runner.generate(ids, { maxNewTokens: 24, eos: t.eos, pick: rarityPicker(t, sampler, 0, index) });
  assert.deepEqual(zero, plain);
});
