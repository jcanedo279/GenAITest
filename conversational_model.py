from transformers import AutoTokenizer, GPTJForCausalLM, StoppingCriteria, StoppingCriteriaList
import torch
import config_util


DEVICE = config_util.GetModelConfig("Device")
MAX_RESPONSE_SIZE = 400
CHARACTER_NAME = "Merlin"
USER_KEY = "<USER>"
USER_NAME = "Jcanedo"
SCENARIO = f"Merlin and {USER_KEY} are on the phone with each other."
EXAMPLE_PROMPT = "What profound lesson or insight do you believe transcends time and holds eternal significance for humanity?"

# Traits such as "Wise"/"Old" are weighted lightly, 'bake' them in by using Mind+Personality.
PERSONA = f'''
    [character("{CHARACTER_NAME}")
    {{
    Species("Human")
    Mind("Wise" + "Old" + "Magical" + "Timeless")
    Personality("Wise" + "Old" + "Magical" + "Timeless")
    Body("152cm tall" + "5 foot tall" + "90 pounds")
    Description("Arthurian legendary wizard" + "Guided Arther to become King" + "advisor for the creation of the Knights of the Round Table")
    Loves("Hares" + "Magic" + "Knowledge" + "adventure partner Nimue")
    Gender("Male" + "Asexual" + "Nonsexual")
    Pronouns("He" + "Him")
    }}]
    '''


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stop_words, tokenizer):
      super().__init__()
      self.stop_words = stop_words
      self.num_tokens_per_stop_word = [len(tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze())
            for stop_word in stop_words]
      self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        gen_tokens = input_ids[0]
        for stop_word, num_tokens in zip(self.stop_words, self.num_tokens_per_stop_word):
            # Multiple tokens can have the same translation so we must decode here :<
            # Also strip because of decoding generating spaces and newlines.
            if stop_word.strip() == self.tokenizer.decode(gen_tokens[-num_tokens:]).strip():
                return True
        return False


class ConversationalModel:
    model_type_to_name = {
        "Pygmalion": "PygmalionAI/pygmalion-6b"
    }

    def make_model_input(prompt):
            return f'''
            {CHARACTER_NAME}'s Persona: {PERSONA}
            Scenario("{SCENARIO}")
            <START>
            You: {prompt}
            {CHARACTER_NAME}:
            '''

    def __init__(self):
        model_name = ConversationalModel.model_type_to_name["Pygmalion"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_input = ConversationalModel.make_model_input(EXAMPLE_PROMPT)

        # We stop generation when the model begins hallucinating the user's response.
        stop_words = ["You:"]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stop_words=stop_words, tokenizer=self.tokenizer)])
        self.model = GPTJForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(DEVICE)

        print('-'*len(EXAMPLE_PROMPT))
        print("Conversation with 'Merlin':")
        print(f"You: {EXAMPLE_PROMPT}")
        self.generate_response(model_input, stopping_criteria)

        while True:
            user_input = input("You: ")
            model_input = ConversationalModel.make_model_input(user_input)
            self.generate_response(model_input, stopping_criteria)
            

    def generate_response(self, model_input, stopping_criteria):
        model_input = ( model_input )
        context_ids = self.tokenizer(model_input, return_tensors="pt").input_ids.to(DEVICE)
        gen_tokens = self.model.generate(
            context_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.9,
            max_new_tokens=MAX_RESPONSE_SIZE,
            stopping_criteria=stopping_criteria
        )
        gen_text = self.tokenizer.batch_decode(gen_tokens)[0]

        # We want to text between '[CHARACTER]' (where the response is) and 'You:' (in our stopping_criteria).
        gen_response = gen_text.split(f"{CHARACTER_NAME}:")[-1].split("You:")[0]
        gen_response.replace("<|endoftext|>", "")
        gen_response.replace(USER_KEY, USER_NAME)
        # Strip a bunch of misc things such as newlines, spacing, and junk from dailog format and model.
        gen_response.strip(':\n\"')
        print(repr(f"{CHARACTER_NAME}: {gen_response}"))


def runner():
    ConversationalModel()

if __name__ == "__main__":
    runner()
