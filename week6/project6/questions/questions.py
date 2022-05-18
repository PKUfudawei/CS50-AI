from re import A
import nltk
import sys
import os
import string
import numpy as np

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    content={}
    for dir in os.listdir(directory):
        if dir.endswith(".txt"):
            name = dir.split(".")[0]
            with open(os.path.join(directory, dir), encoding="utf-8") as f:
                content[name]=f.read()
    return content
        


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words=[w for w in nltk.word_tokenize(document.lower())
           if (w not in string.punctuation) and 
           (w not in nltk.corpus.stopwords.words("english"))]
    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    num_doc = len(documents)
    idf={
        w: np.log(
            num_doc / sum(w in d for d in documents.values())
        ) for w in set().union(*documents.values()) ## or set(sum(documents.values(), []))
    }
    return idf
    


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tf_idf_sum={
        f: sum(
            (idfs[w]*files[f].count(w) if w in files[f] else 0) for w in query 
        ) for f in files
    }
    most_match=sorted(tf_idf_sum, key=tf_idf_sum.get, reverse=True)
    return most_match[:n]
    

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    idf_sum_qtd={
        s: (
            sum((idfs[w] if w in sentences[s] else 0) for w in query), 
            len(set(sentences[s]).intersection(query))/len(sentences[s])
        ) for s in sentences
    }
    most_match=sorted(idf_sum_qtd, key=idf_sum_qtd.get, reverse=True)
    return most_match[:n]


if __name__ == "__main__":
    main()
