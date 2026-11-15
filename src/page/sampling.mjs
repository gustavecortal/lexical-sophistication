// Throws on a sampler the decode loop cannot use (checked before any token).
export function validateSampler(s) {
  if (!(s.temperature > 0 && Number.isFinite(s.temperature))) throw new Error("temperature must be a positive number");
  if (!(s.top_p > 0 && s.top_p <= 1)) throw new Error("top-p must be in (0, 1]");
  if (!(Number.isInteger(s.top_k) && s.top_k >= 1 && s.top_k <= 1000)) throw new Error("top-k must be an integer from 1 to 1000");
  if (!(Number.isFinite(s.presence_penalty) && Math.abs(s.presence_penalty) <= 2)) throw new Error("presence penalty must be from -2 to 2");
  if (!Number.isInteger(s.seed)) throw new Error("seed must be an integer");
}

// Greedy, or the presence penalty → temperature → top-k → top-p → a seeded draw (llama.cpp's order).
export function picker(sampler) {
  if (!sampler) return argmax;
  const seen = new Set();
  let seed = sampler.seed >>> 0;
  const random = () => {  // mulberry32
    seed = (seed + 0x6d2b79f5) >>> 0;
    let t = seed;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return (logits) => {
    for (const token of seen) logits[token] -= sampler.presence_penalty;  // in place: the next step rewrites logits
    const top = [];  // the k largest, descending
    for (let i = 0; i < logits.length; i++) {
      const v = logits[i];
      if (top.length === sampler.top_k && v <= top[top.length - 1][1]) continue;
      let j = top.length === sampler.top_k ? top.length - 1 : top.length;
      while (j > 0 && top[j - 1][1] < v) { top[j] = top[j - 1]; j--; }
      top[j] = [i, v];
    }
    const weights = top.map(([, v]) => Math.exp((v - top[0][1]) / sampler.temperature));
    const total = weights.reduce((a, b) => a + b, 0);
    let kept = 0, mass = 0;
    while (kept < top.length && mass < sampler.top_p) mass += weights[kept++] / total;
    let r = random() * mass, i = 0;
    while (i < kept - 1 && (r -= weights[i] / total) > 0) i++;
    const token = top[i][0];
    seen.add(token);
    return token;
  };
}

function argmax(logits) {
  let best = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
  return best;
}

