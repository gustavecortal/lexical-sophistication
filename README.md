# lexical sophistication

[lexical sophistication](https://gustavecortal.com/lexical-sophistication/) is a sampler that steers a language model toward common or rare words by how often they appear in English. This repository runs it from the command line on Qwen3.5-0.8B, the model of the [live demo](https://gustavecortal.com/lexical-sophistication/).

<p align="center">
  <a href="https://gustavecortal.com/lexical-sophistication/"><img src="docs/demo.webp" width="720" alt="The same message answered with the slider at rare words (+3), then at common words (-3), each rare word shaded"></a>
</p>

## Quick start

```bash
git clone https://github.com/gustavecortal/lexical-sophistication.git
cd lexical-sophistication
npm ci
npx lexical-sophistication "Describe a walk through a forest after the rain." --strength 3
```

The first run downloads the demo's 4-bit ONNX export of the model (0.58 GB) into `~/.cache/lexical-sophistication`, unless the Hugging Face cache holds it. Replies then run on the CPU, at about 15 tokens a second on 16 threads. `--strength` goes from -3 (common words) to 3 (rare words), and 0 gives the model's own reply. The other options (`--seed`, `--max-tokens`, `--temperature`, `--top-k`, `--top-p`, `--presence`) default to the demo's settings.

## How it works

Each word gets a rarity from its frequency in [wordfreq](https://github.com/rspeer/wordfreq)'s English list, 6 minus its Zipf frequency: 0 for words more common than 1 in 1,000, then one point per factor of ten, up to 4. Words of rarity 0 are left alone. Every other word scores its distance from the mean rarity of those words (1.28, weighted by frequency), scaled by the strength. At each step, the model's most likely tokens (20 by default) are raised or lowered by how much they would change the reply's score, and the model's own sampler draws from them. A token that only starts a word counts as the mean score of the words it could become, weighted by their frequency.

## Rebuild the word list

```bash
uv run scripts/export_frequencies.py data/frequency-en.json
node scripts/build_table.mjs data/frequency-en.json data/frequency-en.bin
```

This rewrites `data/frequency-en.bin` from wordfreq 3.1.1 in under a minute, identical to the live demo's table. `npm test` checks the sampler, the model's next tokens and a 48-token reply at strength 3 against the demo.

## License

The code is under the [MIT license](LICENSE). `data/frequency-en.bin` is derived from wordfreq's data by Robyn Speer, under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
