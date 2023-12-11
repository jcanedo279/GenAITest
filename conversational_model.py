from transformers import AutoTokenizer, GPTJForCausalLM, StoppingCriteria, StoppingCriteriaList
import torch
import config_util
# from timeit import default_timer as timer


DEVICE = config_util.GetModelConfig("Device")

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stop_token_id):
      super().__init__()
      self.stop_token_id = stop_token_id.to(DEVICE)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if (len(input_ids[0]) > len(self.stop_token_id)):
            last_output_id = input_ids[0][-len(self.stop_token_id):]
            if torch.equal(self.stop_token_id, last_output_id):
                return True
        return False

class ConversationalModel:
    model_type_to_name = {
        "Pygmalion": "PygmalionAI/pygmalion-6b"
    }

    def __init__(self):
        model_name = ConversationalModel.model_type_to_name["Pygmalion"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        stop_token_id = tokenizer("END_OF_DIALOG", return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze()
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stop_token_id=stop_token_id)])
        model = GPTJForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(DEVICE)

        context = (
            "[CHARACTER]'s Persona: [A few sentences about the character you want the model to play]"
            "Merlin, egendary wizard, embodies enigmatic wisdom and magical prowess, timeless aura, bridges the realms of mysticism and sagacity, old, wise"
            "[DIALOGUE HISTORY]"
            "You: What profound lesson or insight do you believe transcends time and holds eternal significance for humanity?"
            "[CHARACTER]:")
        context_ids = tokenizer(context, return_tensors="pt").input_ids.to(DEVICE)
        
        gen_tokens = model.generate(
            context_ids,
            do_sample=True,
            temperature=0.9,
            max_length=2048,
            stopping_criteria=stopping_criteria
        )
        gen_text = tokenizer.batch_decode(gen_tokens)
        print(gen_text)



def runner():
    convy = ConversationalModel()

if __name__ == "__main__":
    runner()
