// The model's tokenizer and chat template in JS: the same files and rendering as workbench.chat, so the page,
// the Python fit and the hosted fit build identical prompt IDs (check.mjs compares them).
import { Tokenizer } from "@huggingface/tokenizers";
import { Template } from "@huggingface/jinja";

// A thinking prompt ends with THINK, so a reply made with thinking is kept as THINK + its text: the template then shows
// later prompts only its answer, after THINK_END.
export const THINK = "<think>\n", THINK_END = "</think>";
// Fed when a reply's reasoning reaches its budget: a general sentence, as chat.py's.
const CLOSE_THINKING = `\n\nI'll stop thinking here and answer now.\n${THINK_END}\n\n`;

export const sha256 = async (bytes) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))]
  .map((b) => b.toString(16).padStart(2, "0")).join("");

// read(name) gives a file of the model repository as text: fetch in the page, the HF cache in node.
export async function loadTokenizer(read) {
  const [tokenizerText, configText, templateText] = await Promise.all([
    read("tokenizer.json"), read("tokenizer_config.json"), read("chat_template.jinja").catch(() => null)]);
  const json = JSON.parse(tokenizerText), config = JSON.parse(configText);
  const source = templateText ?? config.chat_template;
  const fast = new Tokenizer(json, config);
  const template = new Template(source);
  // Like transformers: the special-token strings are template variables.
  const special = Object.fromEntries(Object.entries(config).filter(([k, v]) => k.endsWith("_token") && typeof v === "string"));
  const encode = (text) => fast.encode(text, { add_special_tokens: false }).ids;
  // A reply stopped while thinking has no answer: closed here, so the template does not show its reasoning as one.
  const closed = (m) => m.role === "assistant" && m.content.startsWith(THINK) && !m.content.includes(THINK_END)
    ? { ...m, content: m.content + THINK_END } : m;
  return {
    eos: encode(config.eos_token)[0],
    thinkEnd: encode(THINK_END)[0], closeThinking: encode(CLOSE_THINKING),  // for a reasoning budget
    encode,
    decode: (ids) => fast.decode(ids, { skip_special_tokens: true }),
    // IDs of the chat prefix ending with the generation prompt.
    prompt: (messages, thinking = false) =>
      encode(template.render({ messages: messages.map(closed), add_generation_prompt: true, enable_thinking: thinking, ...special })),
  };
}
