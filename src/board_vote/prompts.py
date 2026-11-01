"""Module 1b: turn an Item into the exact chat messages and prompt text sent to the model.

Input:  an Item and the `prompt` section of the protocol config.
Output: messages = [{"role": "system", ...}, {"role": "user", ...}]; prompt text = the model's chat
        template applied to them, ending with the assistant turn opener (add_generation_prompt).
Options keep the dataset order (no shuffling), as the protocol says.
"""
from .data import LETTERS, Item


def build_messages(item: Item, prompt_cfg: dict) -> list[dict]:
    fields = {"question": item.question}
    fields.update({f"option_{L.lower()}": text for L, text in zip(LETTERS, item.options)})
    user = prompt_cfg["user_template"].format(**fields)
    return [{"role": "system", "content": prompt_cfg["system"]}, {"role": "user", "content": user}]


def render_prompt(messages: list[dict], tokenizer) -> str:
    """Prompt text for the engine. `tokenizer` is the model's Hugging Face tokenizer (or a test stub)."""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
