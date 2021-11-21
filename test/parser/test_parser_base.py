from pytest import fixture

from parser.parser_base import ParserBase


class TestParserBase:

    doc1 = "This is a function to test document path similarity."
    doc2 = "Use this function to see if your code in doc_to_synsets and similarity_score is correct!"
    story = (
        'A number of Santander cash machines have been shut down due to "potential criminal activity and vandalism", '
        'the bank has said. Lancashire Police Santander said "five ATMs were shut down immediately" but cash machines '
        'elsewhere remained "fully operational". Police in Cheshire said some ATMs were vandalised, but were not '
        'fitted with "skimming devices". People who have lost money have been urged to contact the bank and police. '
        'Lancashire Police said: "We are still taking reports but a series of similar incidents last week occurred in '
        'all parts of the county. The advice, therefore, is not limited to a specific area or town. "We are advising '
        "the public to be vigilant, in particular of Santander machines, but of any cash machines. Report anything "
        'suspicious, have a visual check of the cash point and if in doubt leave it and go somewhere else." Cheshire '
        'Police said on Monday: "We can confirm that there had been a number of ATMs tampered with over the weekend, '
        'however these are not thought to be linked to skimming devices. "It appeared that in one case keypad numbers '
        "had been stuck down in some way and a ruler had been attached to anther causing an issue with the buttons. "
        '"[The incidents] were confined to the eastern part of the county." Please include a contact number if you '
        "are willing to speak to a BBC journalist. You can also contact us in the following ways: Or use the form "
        "below If you are happy to be contacted by a BBC journalist please leave a telephone number that we can "
        "contact you on. In some cases a selection of your comments will be published, displaying your name as you "
        "provide it and location, unless you state otherwise. Your contact details will never be published. When "
        "sending us pictures, video or eyewitness accounts at no time should you endanger yourself or others, "
        "take any unnecessary risks or infringe any laws. Please ensure you have read the terms and conditions. "
    )

    @fixture
    def instance(self):
        return ParserBase(document=self.doc1)

    @fixture
    def instance_story(self):
        return ParserBase(document=self.story)

    def test_tokenize(self, instance):
        assert instance.tokenize() == [["This", "is", "a", "function", "to", "test", "document", "path", "similarity", "."]]
        assert instance.tokenize(include_stop_words=False) == [["This", "function", "test", "document", "path", "similarity", "."]]
        assert instance.tokenize(include_stop_words=False, include_punctuation=False) == [
            ["This", "function", "test", "document", "path", "similarity"]
        ]

    def test_tokenize__more_sentences(self, instance_story):
        assert len(instance_story.tokenize()) == 17
        assert instance_story.tokenize(include_stop_words=False, include_punctuation=False)[0] == [
            "A",
            "number",
            "Santander",
            "cash",
            "machines",
            "shut",
            "due",
            "potential",
            "criminal",
            "activity",
            "vandalism",
            "bank",
            "said",
        ]
