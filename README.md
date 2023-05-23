# Noolp
Natural Language Processing library.

## Dependencies
Here the libraries used this project:
- Python >= 3.7
- [numpy](https://pypi.python.org/pypi/numpy) 
- [sklearn](http://scikit-learn.org/stable/install.html)
- [gensim](https://github.com/RaRe-Technologies/gensim)
- [nltk](http://www.nltk.org)


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

