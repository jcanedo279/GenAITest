from transformers import AutoTokenizer, GPTJForCausalLM
from transformers import logging
import torch
from pathlib import Path
import sys

# Pop up a dir.
sys.path.append(str(Path(__file__).absolute().parent.parent))
# Import from parent directory.
from config_util import GetModelConfig
# Go back to file path.
sys.path.pop()


DEVICE = GetModelConfig("Device")
def main():
    tokenizer = AutoTokenizer.from_pretrained("PygmalionAI/pygmalion-6b")
    model = GPTJForCausalLM.from_pretrained("PygmalionAI/pygmalion-6b", torch_dtype=torch.float16).to(DEVICE)

    example_text = (
        "[CHARACTER]'s Persona: [A few sentences about the character you want the model to play]"
        "Merlin, the legendary wizard, embodies enigmatic wisdom and magical prowess, with a timeless aura that bridges the realms of mysticism and sagacity"
        "[DIALOGUE HISTORY]"
        "You: What profound lesson or insight do you believe transcends time and holds eternal significance for humanity?"
        "[CHARACTER]:")
    input_ids = tokenizer(example_text, return_tensors="pt").input_ids.to(DEVICE)
    
    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.9,
        max_length=300
    )
    gen_text = tokenizer.batch_decode(gen_tokens)
    print(gen_text)

    


if __name__ == "__main__":
    logging.set_verbosity_error()
    main()
