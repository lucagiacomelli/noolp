from typing import Optional

from transformers import pipeline


class Summarizer:
    def __init__(self, document: str, model: str = "t5-small"):
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
        return ""
