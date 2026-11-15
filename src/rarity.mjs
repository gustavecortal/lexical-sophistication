// Word-rarity sampling: biases a language model toward common or rare words of a reference corpus, in the
// prefix-aware form of experiments/lexical-diversity/prefix.mjs (the handoff's reference implementation): each
// candidate token gets the change in text potential, and an unfinished word is valued by the corpus words it can
// still become. Three changes from the handoff, each measured on Qwen3.5-0.8B (WORKING_RECORD.md): words are
// centered on content words, so grammar words keep their probability; hyphens end a word, and digits and unknown
// possessives are scored as wordfreq stores them; strings missing from the corpus are never favored. This version
// also decodes only the text after the last whitespace boundary, so its cost per step does not grow with the reply.
import { picker, validateSampler } from "./page/sampling.mjs";

export const MAX_STRENGTH = 4;

export const WORD = /[\p{L}\p{N}][\p{L}\p{M}\p{N}]*(?:['’][\p{L}\p{N}][\p{L}\p{M}\p{N}]*)*/gu;
export const normalize = (text) => text.normalize("NFC").toLowerCase().replaceAll("’", "'");
const checkStrength = (strength) => {
  if (!(Number.isFinite(strength) && Math.abs(strength) <= MAX_STRENGTH)) throw new Error(`strength must be in [-${MAX_STRENGTH}, ${MAX_STRENGTH}]`);
};

// Surface words, NFC + lowercase, as wordfreq segments English: internal apostrophes stay inside a word, hyphens split.
// Unless final, a trailing word (or one followed only by an apostrophe awaiting its next fragment) is left out.
export function words(text, final = true) {
  const nfc = text.normalize("NFC"), matches = [...nfc.matchAll(WORD)];
  if (!final && matches.length) {
    const last = matches.at(-1);
    if (/^['’�]*$/u.test(nfc.slice(last.index + last[0].length))) matches.pop();
  }
  return matches.map((m) => normalize(m[0]));
}

// One unit per decade below a frequency of 1/1000, capped at four.
export const rarity = (f) => Math.min(4, Math.max(0, -Math.log10(f) - 3));

// Binary table (scripts/build_table.mjs), gzip-compressed: u32 header length, header JSON {language, source, count,
// palette}, u16 palette index per word (aligned to 2), then the words, sorted, front-coded one per line: a character
// 0x20 + the length shared with the previous word, then the rest.
export async function readTable(compressed) {
  const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream("gzip"));
  const bytes = new Uint8Array(await new Response(stream).arrayBuffer());
  const view = new DataView(bytes.buffer), length = view.getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(4, 4 + length)));
  const at = (4 + length + 1) & ~1, palette = new Uint16Array(bytes.buffer.slice(at, at + 2 * header.count));
  const lines = new TextDecoder().decode(bytes.subarray(at + 2 * header.count)).split("\n");
  const keys = new Array(header.count), frequencies = new Float64Array(header.count);
  let previous = "";
  for (let i = 0; i < header.count; i++) {
    const line = lines[i];
    previous = keys[i] = previous.slice(0, line.charCodeAt(0) - 0x20) + line.slice(1);
    frequencies[i] = header.palette[palette[i]];
  }
  return { language: header.language, source: header.source, keys, frequencies };
}

// From {word: frequency}, as the experiments' JSON tables: normalized single words with a frequency in (0, 1].
export function tableFrom(object) {
  const valid = Object.entries(object).filter(([w, f]) => Number.isFinite(f) && f > 0 && f <= 1 && words(w).length === 1 && words(w)[0] === w);
  valid.sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
  return { keys: valid.map(([w]) => w), frequencies: Float64Array.from(valid, ([, f]) => f) };
}

const lowerBound = (keys, value) => {
  let lo = 0, hi = keys.length;
  while (lo < hi) { const mid = (lo + hi) >>> 1; if (keys[mid] < value) lo = mid + 1; else hi = mid; }
  return lo;
};

// A word's score at strength λ, in logit units. Words more common than 1 in 1,000 (rarity 0, mostly grammar words)
// score 0; every other word scores λ(r − r̄), r̄ being those words' corpus-weighted mean rarity, so a text of average
// rarity scores 0 whatever its length. (The handoff scored λr − ln E[exp(λr)], which at λ = 2 cost every common word
// 4 logits: rare mode dropped articles and common mode lengthened replies.) A word with a digit scores 0 (wordfreq
// stores numbers in digit-collapsed bins); an unknown possessive scores as its stem; any other unknown string scores
// the lowest attested score, so a misspelling never beats a real word in either direction.
// An unfinished word p is valued at the mean score of the corpus words starting with p, weighted by frequency: one key
// range of the sorted words, found by two binary searches over cumulative masses. p with no completions takes the
// unknown score. Along any spelling the values telescope to the word's score, however tokens split it, and the mean
// over the whole corpus is 0, so starting a word is neutral on average. (The handoff used ln E[exp(score) | p], which at
// high strength let a rare tail dominate: " sky" rose from 55% to 88% at +3 as a possible "skyscraper", while
// " canopy" fell; with the mean, " canopy" rises from 16% to 52%.) Each strength's masses are built once (at most four
// kept); prefix values are cached per strength.
export function createIndex({ keys, frequencies }) {
  if (!keys.length) throw new Error("no reference words");
  for (let i = 1; i < keys.length; i++) if (!(keys[i - 1] < keys[i])) throw new Error("reference words must be sorted and distinct");
  const find = (w) => { const i = lowerBound(keys, w); return keys[i] === w ? i : -1; };
  // A word's position, or its stem's for a possessive missing from the table; -1 if neither is there.
  const locate = (w) => { const i = find(w); return i < 0 && w.endsWith("'s") ? find(w.slice(0, -2)) : i; };
  const span = (p) => [lowerBound(keys, p), lowerBound(keys, p + "￿")];  // the keys starting with p
  const mass = new Float64Array(keys.length + 1), scores = Float64Array.from(frequencies, rarity);
  let content = 0, rbar = 0;  // corpus mass and mean rarity of the words with rarity above 0
  for (let i = 0; i < keys.length; i++) {
    mass[i + 1] = mass[i] + frequencies[i];
    if (scores[i] > 0) { content += frequencies[i]; rbar += frequencies[i] * scores[i]; }
  }
  rbar /= content;
  const cached = new Map();
  const scorer = (strength) => {
    checkStrength(strength);
    if (cached.has(strength)) return cached.get(strength);
    const values = Float64Array.from(scores, (r) => (r > 0 ? strength * (r - rbar) : 0)), scored = new Float64Array(keys.length + 1);
    let unknown = 0;
    for (let i = 0; i < keys.length; i++) {
      scored[i + 1] = scored[i] + frequencies[i] * values[i];
      unknown = Math.min(unknown, values[i]);
    }
    const prefixes = new Map();
    const word = (raw) => {
      const w = normalize(raw);
      if (/\p{N}/u.test(w)) return 0;
      const i = locate(w);
      return i < 0 ? unknown : values[i];
    };
    const prefix = (raw) => {
      const p = normalize(raw).replace(/�+$/u, "");  // an incomplete character is not judged yet
      if (!p) return 0;
      let value = prefixes.get(p);
      if (value !== undefined) return value;
      if (/\p{N}/u.test(p)) value = 0;
      else {
        const [start, end] = span(p), base = mass[end] - mass[start];
        const stem = p.match(/^(.+)'s?$/u)?.[1];  // a possessive in progress: its stem's score
        value = base > 0 ? (scored[end] - scored[start]) / base : stem ? word(stem) : unknown;
      }
      if (prefixes.size >= 32768) prefixes.clear();
      prefixes.set(p, value);
      return value;
    };
    // Completed words plus the unfinished one.
    const text = (raw, final = false) => {
      const completed = words(raw, final);
      let value = 0;
      for (const w of completed) value += word(w);
      if (!final) {
        const all = words(raw);
        if (all.length > completed.length) value += prefix(all.at(-1) + (normalize(raw).match(/'+$/u)?.[0] ?? ""));
      }
      return value;
    };
    const result = { prefix, word, text };
    if (cached.size >= 4) cached.delete(cached.keys().next().value);
    cached.set(strength, result);
    return result;
  };
  // The corpus words starting with p: how many, their frequency-weighted mean rarity, and the n most frequent (ties in
  // word order), kept by insertion: a one-letter prefix has some 30,000 words, which a full sort took 5 ms to order.
  const completions = (p, n = 6) => {
    const [start, end] = span(p), top = [];
    let total = 0, weighted = 0;
    for (let i = start; i < end; i++) {
      const f = frequencies[i];
      total += f; weighted += f * scores[i];
      if (top.length === n && f <= top[n - 1].f) continue;
      let j = top.length === n ? n - 1 : top.length;
      while (j && top[j - 1].f < f) { top[j] = top[j - 1]; j--; }
      top[j] = { w: keys[i], f };
    }
    return { count: end - start, mean: total ? weighted / total : null, top };
  };
  const at = (i) => (i < 0 ? null : frequencies[i]);
  // frequency: the word's own entry; lookup: with a possessive's stem when the possessive is missing, as the scorer does.
  return { entries: keys.length, rbar, scorer, completions, frequency: (w) => at(find(w)), lookup: (w) => at(locate(w)) };
}

// The model's candidate next tokens after the text `prefix`, for the explanatory figure. rows: [{id, piece, logit}]
// (piece the token's decoded text, appended to prefix; none is the end of the reply). Every score is the strength
// times a fixed number, so a candidate's bonus is λ·up for λ ≥ 0 and |λ|·down for λ < 0 (probabilities() below), with
// up and down its bonuses at 1 and −1. Also the word the token starts or extends, or the one it only ends (`ends`, for
// a piece without letters or digits, as the comma in "narrow,"), its corpus frequency and rarity, the corpus words
// that unfinished word can still become, and `value`: the rarity the sampler treats that word as having, r̄ plus its
// score at strength 1. That is the word's rarity for a content word, r̄ for a word too common to score (rarity 0), and
// for an unfinished word the mean over its completions with those common words at r̄.
export function candidates(index, prefix, rows) {
  const one = index.scorer(1), up = one.text, down = index.scorer(-1).text, upBefore = up(prefix), downBefore = down(prefix);
  const previous = words(prefix, false).length, last = words(prefix).at(-1) ?? null;
  return rows.map(({ piece, logit }) => {
    const after = prefix + piece, completed = words(after, false), all = words(after);
    const pending = all.length > completed.length ? all.at(-1) : null;
    const word = pending ?? completed.at(-1) ?? null, f = word ? index.frequency(word) : null;
    return {
      piece, logit, up: up(after) - upBefore, down: down(after) - downBefore,
      word, ends: !pending && completed.length > previous && !/[\p{L}\p{N}]/u.test(piece) ? last : null,
      frequency: f, rarity: f === null ? null : rarity(f),
      completions: pending ? index.completions(normalize(pending)) : null,
      value: index.rbar + (pending ? one.prefix(pending) : word ? one.word(word) : 0),
    };
  });
}

// The candidates' probabilities at strength s, from the bonuses candidates() gives (logit + λ·up, or |λ|·down).
export function probabilities(tokens, s) {
  const a = tokens.map((t) => t.logit + (s >= 0 ? s * t.up : -s * t.down)), m = Math.max(...a);
  const e = a.map((x) => Math.exp(x - m)), z = e.reduce((x, y) => x + y, 0);
  return e.map((x) => x / z);
}

// A per-reply pick(logits) → token. Strength 0 is the original sampler, draw for draw. Otherwise the presence-adjusted
// model top-k is the support, each candidate's logit gains potential(text after) − potential(text before), and the
// original sampler draws from the result. Words never contain whitespace, so the potential of the whole reply is the
// sum over its whitespace-separated parts: only the tail after the last boundary is decoded and scored. The tail is
// cut before a token only when decoding it separately reproduces the joined text, so byte-split characters and
// tokenizers that drop a leading space keep the tail whole.
export function rarityPicker(tokenizer, sampler, strength, index) {
  validateSampler(sampler);
  checkStrength(strength);
  const pick = picker(sampler);
  if (!strength) return pick;
  const potential = index.scorer(strength).text, decode = (ids) => (ids.length ? tokenizer.decode(ids) : "");
  let seen = null, tail = [], tailText = "", before = 0, adjusted = null;  // seen: generated tokens, as a mask
  return (logits) => {
    seen ??= new Uint8Array(logits.length);
    const top = [];  // presence-adjusted top-k, descending, ties in token order
    for (let id = 0; id < logits.length; id++) {
      const score = logits[id] - (seen[id] ? sampler.presence_penalty : 0);
      if (!Number.isFinite(score)) continue;
      if (top.length === sampler.top_k && score <= top[top.length - 1][1]) continue;
      let i = top.length === sampler.top_k ? top.length - 1 : top.length;
      while (i && top[i - 1][1] < score) { top[i] = top[i - 1]; i--; }
      top[i] = [id, score];
    }
    if (!top.length) throw new Error("no finite candidate logits");
    adjusted ??= new Float32Array(logits.length);
    adjusted.fill(-Infinity);
    for (const [id] of top) adjusted[id] = logits[id] + potential(decode([...tail, id]), id === tokenizer.eos) - before;
    const token = pick(adjusted);
    seen[token] = 1;
    const text = decode([...tail, token]);
    if (/\s$/u.test(text)) [tail, tailText] = [[], ""];
    else {
      const piece = decode([token]);
      if (tail.length && /^\s/u.test(piece) && tailText + piece === text && !tailText.endsWith("�")) [tail, tailText] = [[token], piece];
      else [tail, tailText] = [[...tail, token], text];
    }
    before = potential(tailText, false);
    return token;
  };
}
