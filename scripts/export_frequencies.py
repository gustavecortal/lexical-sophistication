# /// script
# requires-python = ">=3.12"
# dependencies = ["wordfreq==3.1.1"]
# ///
"""Writes wordfreq's large English list as JSON: each word, NFC and lower case, with its frequency.
Usage: uv run scripts/export_frequencies.py data/frequency-en.json"""
import importlib.metadata
import json
import sys
import unicodedata
from pathlib import Path

from wordfreq import iter_wordlist, word_frequency

frequencies = {}
for word in iter_wordlist("en", wordlist="large"):
    normalized = unicodedata.normalize("NFC", word).lower().replace("’", "'")
    frequency = word_frequency(word, "en", wordlist="large")
    if frequency > 0:
        frequencies[normalized] = max(frequencies.get(normalized, 0), frequency)
output = Path(sys.argv[1])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({"language": "en",
    "source": {"name": "wordfreq", "version": importlib.metadata.version("wordfreq"),
               "wordlist": "large", "url": "https://github.com/rspeer/wordfreq"},
    "frequencies": frequencies}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print(f"{len(frequencies)} words written to {output}")
