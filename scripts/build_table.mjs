// Packs the exported word list (export_frequencies.py) into the table the sampler reads (src/rarity.mjs readTable).
// Usage: node scripts/build_table.mjs data/frequency-en.json data/frequency-en.bin
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { gzipSync } from "node:zlib";
import { tableFrom } from "../src/rarity.mjs";

const [input, output] = process.argv.slice(2);
const json = JSON.parse(readFileSync(input, "utf8")), { keys, frequencies } = tableFrom(json.frequencies);
const palette = [...new Set(frequencies)].sort((a, b) => b - a);
if (palette.length > 65536) throw new Error("too many distinct frequencies for a 16-bit palette");
const position = new Map(palette.map((f, i) => [f, i]));
let lines = "", previous = "";
for (const word of keys) {
  let shared = 0;
  while (shared < 95 && shared < previous.length && shared < word.length && previous[shared] === word[shared]) shared++;
  if (shared && /[\uD800-\uDBFF]/.test(word[shared - 1])) shared--;  // never split a surrogate pair
  lines += String.fromCharCode(0x20 + shared) + word.slice(shared) + "\n";
  previous = word;
}
const header = Buffer.from(JSON.stringify({ language: json.language, source: json.source, count: keys.length, palette }));
const at = (4 + header.length + 1) & ~1, body = Buffer.alloc(at + 2 * keys.length);
body.writeUInt32LE(header.length, 0);
header.copy(body, 4);
Buffer.from(Uint16Array.from(frequencies, (f) => position.get(f)).buffer).copy(body, at);
const packed = gzipSync(Buffer.concat([body, Buffer.from(lines, "utf8")]), { level: 9 });
mkdirSync(dirname(output), { recursive: true });
writeFileSync(output, packed);
console.log(`${keys.length} words, ${palette.length} distinct frequencies, ${packed.length} bytes → ${output}`);
