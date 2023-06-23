from typing import Optional

from transformers import pipeline


class Summarizer:
    def __init__(self, document: str, model: str):
        self.document = document
        self.model = model

    def summarize(
        self,
        use_pipeline: bool = True,
        auto: bool = True,
        min_length: int = 20,
        max_length: int = 60,
    ) -> Optional[str]:
        """Using the Hagging face pipeline summarization task, it returns a summary of the document"""

        if use_pipeline:
            summarizer = pipeline(
                task="summarization",
                model=self.model,
                min_length=min_length,
                max_length=max_length,
                truncation=True,
                # model_kwargs={"cache_dir": ""},
            )
            summaries = summarizer(self.document)
            return summaries[0]["summary_text"] if summaries else None

        if auto:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            tokenizer = AutoTokenizer.from_pretrained("t5-small", cache_dir=None)
            model = AutoModelForSeq2SeqLM.from_pretrained("t5-small", cache_dir=None)
        else:
            from transformers import T5Tokenizer, T5ForConditionalGeneration

            tokenizer = T5Tokenizer.from_pretrained("t5-small", cache_dir=None)
            model = T5ForConditionalGeneration.from_pretrained(
                "t5-small", cache_dir=None
            )

        # For summarization, T5-small expects a prefix "summarize: ", so we prepend that to each article as a prompt.
        articles = [self.document]

        # Tokenize the input
        inputs = tokenizer(
            articles,
            max_length=1024,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        summary_ids = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            num_beams=2,
            min_length=0,
            max_length=40,
        )
        decoded_summaries = tokenizer.batch_decode(
            summary_ids, skip_special_tokens=True
        )
        return decoded_summaries[0] if decoded_summaries else None

    def few_shot_pipeline(self):
        """TODO: move this code elsewhere"""

        few_shot_pipeline = pipeline(
            task="text-generation",
            model="EleutherAI/gpt-neo-1.3B",
            max_new_tokens=10,
            # model_kwargs={"cache_dir": ""},
        )

        # Get the token ID for "###", which we will use as the EOS token below.
        eos_token_id = few_shot_pipeline.tokenizer.encode("###")[0]

        results = few_shot_pipeline(
            """For each tweet, describe its sentiment:

        [Tweet]: "I hate it when my phone battery dies."
        [Sentiment]: Negative
        ###
        [Tweet]: "My day has been 👍"
        [Sentiment]: Positive
        ###
        [Tweet]: "This is the link to the article"
        [Sentiment]: Neutral
        ###
        [Tweet]: "This new music video was incredible"
        [Sentiment]:""",
            eos_token_id=eos_token_id,
        )

        print(results[0]["generated_text"])

        # We override max_new_tokens to generate longer answers.
        # These book descriptions were taken from their corresponding Wikipedia pages.
        results = few_shot_pipeline(
            """Generate a book summary from the title:
        
        [book title]: "Stranger in a Strange Land"
        [book description]: "This novel tells the story of Valentine Michael Smith, a human who comes to Earth in early adulthood after being born on the planet Mars and raised by Martians, and explores his interaction with and eventual transformation of Terran culture."
        ###
        [book title]: "The Adventures of Tom Sawyer"
        [book description]: "This novel is about a boy growing up along the Mississippi River. It is set in the 1840s in the town of St. Petersburg, which is based on Hannibal, Missouri, where Twain lived as a boy. In the novel, Tom Sawyer has several adventures, often with his friend Huckleberry Finn."
        ###
        [book title]: "Dune"
        [book description]: "This novel is set in the distant future amidst a feudal interstellar society in which various noble houses control planetary fiefs. It tells the story of young Paul Atreides, whose family accepts the stewardship of the planet Arrakis. While the planet is an inhospitable and sparsely populated desert wasteland, it is the only source of melange, or spice, a drug that extends life and enhances mental abilities.  The story explores the multilayered interactions of politics, religion, ecology, technology, and human emotion, as the factions of the empire confront each other in a struggle for the control of Arrakis and its spice."
        ###
        [book title]: "Blue Mars"
        [book description]:""",
            eos_token_id=eos_token_id,
            max_new_tokens=50,
        )

        """Generate a rhyme for an existing word. The output should be formatted as JSON that satisfies the JSON schema below.

        For example, for the schema {"properties": {"initial_word": {"type": "string"}, "rhyme_word": {"type": "string"}}, "required": ["initial_word", "rhyme_word"]}}, the object {"initial_word": "bar", "rhyme_word": "foo"} is a well-formatted instance of the schema. The object {"initial_word": "bar"} is not well-formatted.

        [Word]: "Prime"
        [Rhyme JSON]: "{"initial_word": "Prime", "rhyme_word": "Daytime"}"
        ###
        [Word]: "Example"
        [Rhyme JSON]: "{"initial_word": "Example", "rhyme_word": "Cample"}"
        ###
        [Word]: "Chair"
        [Rhyme JSON]: "{"initial_word": "Chair", "rhyme_word": "Share"}"
        ###
        [Word]: "Computer"
        [Rhyme JSON]: """

        print(results[0]["generated_text"])

        return ""
