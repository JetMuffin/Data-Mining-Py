
import os
import re
import traceback
import math
import nltk
from optparse import OptionParser
from stemming.porter2 import stem
import nltk.data

__author__ = "jeff"

stopwords = []
tokenizer = nltk.data.load("")
tokenizer.todchen

class Context(object):
    """
    Context to save all global variables.
    """
    def __init__(self):
        self.documents = []
        self.queue_size = 0
        self.tasks = []
        self.terms = {}
        self.terms_with_id = {}

    def tfidf(self, input):
        # Walk through input directory and get all input documents
        for dir in os.listdir(input):
            for file in os.listdir(os.path.join(input, dir)):
                self.tasks.append((dir, file, os.path.join(os.path.join(input, dir), file)))

        # Tokenize and parse all documents
        for index, (category, document, path) in enumerate(self.tasks):
            print "{document} [{category}] ({now}/{total})".format(category=category,
                                                                 document=document,
                                                                 now=index+1,
                                                                 total=len(self.tasks))
            document = Document(document, category, path)
            self.documents.append(document)
            self.add_terms(document.terms)

        # Update global terms dict with generated id
        self.terms_with_id = {term: index for index, term in enumerate(sorted(self.terms.keys()))}
        print "Get {} terms..".format(len(self.terms_with_id))

        # Update `idf` of every term of every documents after all documents parsed
        for document in self.documents:
            for word, term in document.terms.items():
                term.df = self.terms[term.word]
                term.tfidf = float(term.tf) * float(math.log10(len(self.documents) / term.df))

    def add_terms(self, terms):
        """
        Add term to global context and count the `df` of it.
        Note that this method will be called only once for a document.
        """
        for word in terms.keys():
            if word in self.terms:
                self.terms[word] += 1
            else:
                self.terms[word] = 1

    def generate_vector(self):
        """
        Generate feature vector of all documents.
        Return a list of tuple of first element is the feature list, second element is the category of document.
        """
        sorted_documents = sorted(self.documents, key=lambda x: (x.category, x.name))
        sparse_feature = []

        for document in sorted_documents:
            document_feature = []
            for word, term in document.terms.items():
                if term.tfidf > 0:
                    document_feature.append((self.terms_with_id[word], term.tfidf))
            document_feature.sort(key=lambda x: x[0])
            sparse_feature.append((document_feature, document.category))

        return sparse_feature

    def write_words(self, output):
        """
        Write all context terms to a file.
        """
        print "Write term words to path {}...".format(output)
        sorted_list = sorted(self.terms_with_id.iteritems(), key=lambda d:d[1])
        with open(output, "w") as f:
            for word in sorted_list:
                f.write('{word} {index}\n'.format(word=word[0], index=word[1]))

    def write_features(self, output, category=None):
        """
        Write feature vectors to a file in standard format.
        """
        print "Write features to path {}...".format(output)
        with open(output, "w") as f:
            for index, vector in enumerate(self.generate_vector()):
                if category and category != vector[1]:
                    continue
                f.write(', '.join(['{}:{}'.format(t[0], t[1]) for t in vector[0]]))
                f.write('\n')


class Document(object):

    def __init__(self, name, category, path):
        self.name = name
        self.path = path
        self.category = category
        self.terms = {}
        self.total_term_size = 0
        self._raw_content = self.read()
        self.parse(self._raw_content)

    def read(self):
        """
        Read raw content from file
        """
        with open(self.path) as f:
            text = f.read()
        return text

    def is_word(self, term):
        """
        Determine whether a string is a word or not.
        """
        # pattern = r'^[A-Za-z]+$|\w+(-\w+)'
        pattern = r'^[A-Za-z]+$'
        return re.match(pattern, term)

    def add_term(self, term):
        """
        Add a term to document's term list.
        """
        self.total_term_size += 1
        if term in self.terms:
            self.terms[term].tf += 1
        else:
            self.terms[term] = Term(term, self.name)

    def parse(self, text):
        """
        Tokenize and stem raw text and get the words, using nltk.
        """
        try:
            tokens = nltk.word_tokenize(text.decode('utf-8'))
        except Exception as e:
            traceback.print_exc(e)
            tokens = []

        for word in tokens:
            word = stem(word.lower())
            if self.is_word(word) and word not in stopwords:
                self.add_term(word)

        for word, term in self.terms.items():
            self.terms[word].tf = float(term.tf) / float(self.total_term_size)

    @property
    def term_size(self):
        return len(self.terms)


class Term(object):

    def __init__(self, word, document):
        self.word = word
        self.document = document
        self.tf = 1
        self.idf = self.tfidf = self.id = 0

    def __repr__(self):
        return "{word}(id:{id}, tf: {tf}, idf: {idf}, tfidf:{tfidf})".format(word=self.word,
                                                                             id=self.id,
                                                                             tf=self.tf,
                                                                             idf=self.idf,
                                                                             tfidf=self.tfidf)


def read_stopwords(path):
    global stopwords
    with open(path) as f:
        stopwords = [line.strip() for line in f]

def main():
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input", default="data", help="input directory, `./data` by default")
    parser.add_option("-o", "--output", dest="output", default="result.txt", help="output path, `./result.txt` by default")
    parser.add_option("-c", "--category", dest="category", default=None, help="which category to print, all by default")

    (options, args) = parser.parse_args()
    read_stopwords("stopwords")

    context = Context()
    context.tfidf(options.input)
    context.write_words("words")
    context.write_features(options.output, options.category)

if __name__ == '__main__':
    main()
