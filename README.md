# Noolp
Natural Language Processing library. This library allows to resolve NLP problems using the state-of-the-art approaches.
In particular, Noolp allows to execute the following tasks:
- Simple Parsing and tokenization
- Document similarity
- Summarization
- Topic modelling
- Zero-shot classification

## Dependencies
Check the `requirements.txt` for all the dependencies of this library.

## Setup
Install the dependencies in the virtual environment.
```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python -m nltk.downloader popular
```


## Run the example 
To run the script clone the repo and:
```
python3 main.py
```

## Run the tests
To run the tests install the test requirements first, and then run pytest:
```
pip3 install -r requirements-test.txt
pytest 
```

