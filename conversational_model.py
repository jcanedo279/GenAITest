from transformers import AutoTokenizer, GPTJForCausalLM, StoppingCriteria, StoppingCriteriaList
import torch
import config_util


DEVICE = config_util.GetModelConfig("Device")
MAX_RESPONSE_SIZE = 400
USER_KEY = "<USER>"
USER_NAME = "Jcanedo"

CHARACTER_NAME = "Merlin"
CHARACTER_SPECIES = ["Human"]
CHARACTER_PERSONALITY = ["Wise", "Old", "Magical", "Timeless"]
CHARACTER_BODY = ["152cm tall", "5 foot tall", "90 pounds"]
CHARACTER_DESCRIPTIONS = ["Arthurian legendary wizard", "Guided Arther to become King", "advisor for the creation of the Knights of the Round Table"]
CHARACTER_LOVES = ["Hares", "Magic", "Knowledge", "adventure partner Nimue"]
CHARACTER_GENDER = ["Male", "Asexual", "Nonsexual"]
CHARACTER_PRONOUNS = ["He", "Him"]

SCENARIO = f"Merlin and {USER_KEY} are texting each other..."
STARTING_PROMPT = "What profound lesson or insight do you believe transcends time and holds eternal significance for humanity?"

def PERSONA():
    def wpp_ize_words(words):
        return " + ".join([f"\"{word}\"" for word in words])
    # Personality traits such as "Wise"/"Old" are weighted lightly, 'bake' them in by using Mind+Personality.
    return f'''
        [character("{CHARACTER_NAME}")
        {{
        Species({wpp_ize_words(CHARACTER_SPECIES)})
        Mind({wpp_ize_words(CHARACTER_PERSONALITY)})
        Personality({wpp_ize_words(CHARACTER_PERSONALITY)})
        Body({wpp_ize_words(CHARACTER_BODY)})
        Description({wpp_ize_words(CHARACTER_DESCRIPTIONS)})
        Loves({wpp_ize_words(CHARACTER_LOVES)})
        Gender({wpp_ize_words(CHARACTER_GENDER)})
        Pronouns({wpp_ize_words(CHARACTER_PRONOUNS)})
        }}]'''

def CHARACTER_MODEL_INPUT(prompt):
    return f'''
        {CHARACTER_NAME}'s Persona: {PERSONA()}
        Scenario("{SCENARIO}")
        <START>
        You: {prompt}
        {CHARACTER_NAME}:
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

    def __init__(self):
        model_name = ConversationalModel.model_type_to_name["Pygmalion"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_input = CHARACTER_MODEL_INPUT(STARTING_PROMPT)

        # We stop generation when the model begins hallucinating the user's response.
        stop_words = ["You:"]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stop_words=stop_words, tokenizer=self.tokenizer)])
        self.model = GPTJForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(DEVICE)

        print('-'*len(STARTING_PROMPT))
        print("Conversation with 'Merlin':")
        print(f"You: {STARTING_PROMPT}")
        self.generate_response(model_input, stopping_criteria)

        while True:
            user_input = input("You: ")
            model_input = CHARACTER_MODEL_INPUT(user_input)
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
        # Replace all instances of USER_KEY with USER_NAME.
        USER_NAME.join(gen_response.split(USER_KEY))
        # Strip a bunch of misc things such as newlines, spacing, and junk from dailog format and model.
        garbage_chars = ' :\n\"'
        gen_response_lines = [response_line.strip(garbage_chars) for response_line in gen_response.strip(garbage_chars).split('\n')]
        gen_response = "\n  ".join(gen_response_lines)

        print(f"{CHARACTER_NAME}: {gen_response}")


def runner():
    ConversationalModel()

if __name__ == "__main__":
    runner()
