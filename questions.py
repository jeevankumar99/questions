import nltk
import sys
import os
import string
import math

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
    # dict containg filenames as keys and file content as values
    file_map = {}
    for docs in os.listdir(directory):
        current_file = open(directory + os.sep + docs)
        file_map[docs] = current_file.read()

    return file_map


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    
    tokenized  = nltk.tokenize.word_tokenize(document)
    # nltk stopwords and standard punctuations 
    stopwords = nltk.corpus.stopwords.words("english")
    punctuations = string.punctuation
    
    # filter stopwords and punctuations
    remove_words = []
    for i in range(len(tokenized)):
        tokenized[i] = tokenized[i].lower()
        if  (tokenized[i] in stopwords) or (tokenized[i] in punctuations):
            remove_words.append(tokenized[i])
    
    for words in remove_words:
        tokenized.remove(words)
    
    return tokenized

def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # temp dict to keep word count, and idf for idf value
    temp_idf, idf_dict = {}, {}
    # total docs in corpus
    doc_num = len(documents)
    
    for docs in documents.keys():
        for word in documents[docs]:
            
            if word not in temp_idf.keys():
                temp_idf[word] = [1, [docs], 0]
                # log(doc_num/1)
                idf_dict[word] = math.log(doc_num)
            
            # to filter words repeated in same document
            if docs not in temp_idf[word][1]: 
                temp_idf[word][0] += 1
                # calculate the idf for each word
                idf_dict[word] = math.log(doc_num/temp_idf[word][0])
                temp_idf[word][1].append(docs)
    
    return idf_dict

                 


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tf_idfs = {}
    for doc in files:
        tf_idfs[doc] = 0
        for word in query:
            
            # if word is not in idfs, ignore it
            try:
                temp = idfs[word]
            except KeyError:
                continue
            
            # to get term frequency of a word in a file
            temp_tf = files[doc].count(word)
            
            # tf-idfs = tf * idfs
            tf_idfs[doc] += temp_tf * idfs[word]
    
    # sorting dict based on tf_idfs
    sorted_tf_idfs = sorted(tf_idfs.items(), key = lambda x: x[1], reverse=True)
    sorted_tf_idfs = [docs[0] for docs in sorted_tf_idfs]
    
    # returns n top_files back
    return sorted_tf_idfs[:n]
    

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    top_sent = {}
    for sentence in sentences:
        top_sent[sentence] = [0, 0]
        for word in sentences[sentence]:
            # to get matching words in query and sentence.
            if word in query:
                top_sent[sentence][0] += 1
                try:
                    temp = idfs[word]
                except KeyError:
                    continue
                
                # sum of idfs
                top_sent[sentence][1] += idfs[word]
        
        # to get the query term density
        top_sent[sentence][0] /= len(sentence)
    
    # sort dict based on idfs and query term density
    sorted_top_sent = sorted(top_sent.items(), key = lambda x: (x[1][1], x[1][0]), reverse = True) 
    sorted_top_sent = [sent[0] for sent in sorted_top_sent]
    
    # returns n top sentences
    return (sorted_top_sent[:n])



if __name__ == "__main__":
    main()
