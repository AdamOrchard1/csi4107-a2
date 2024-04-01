import os
from bs4 import BeautifulSoup
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Load spaCy Model
nlp = spacy.load("en_core_web_sm")

# Increase max_length limit
nlp.max_length = 2000000  # Adjust the limit of document size


document_strings = {}
# Directory
directory = 'C:/Users/adamo/OneDrive/Desktop/skool/4107/a2/coll'

# Inverted index as a dictionary
inverted_index = {}

document_ids = []

topic_id = 1

# Define Preprocessing Function using spaCy
def preprocess_spacy(doc_string):
    # Tokenization using spaCy, convert to lowercase
    doc = nlp(doc_string.lower())

    # TODO: come back and make it more readable
    # Lemmatization and remove stopwords and put tokens into array
    tokens = [token.lemma_ for token in doc if token.is_alpha and not nlp.vocab[token.text].is_stop]

    return tokens

def process_document(content):
    soup = BeautifulSoup(content, "html.parser")
    doc_infos = {}

    # Find all DOC tags
    doc_tags = soup.find_all("doc")

    for doc_tag in doc_tags:
        doc_string = ""

        # Extract DOCNO and use it as part of the filename
        docno_tag = doc_tag.find("docno")
        if docno_tag:
            docno = docno_tag.text.strip()
            #TODO: remove this
            # print("Processing Document:", docno)

            # Concatenate the content of relevant tags (HEAD and text)
            head_tag = doc_tag.find("head")
            text_tag = doc_tag.find("text")

            if head_tag:
                doc_string += head_tag.text.strip() + " "
            if text_tag:
                doc_string += text_tag.text.strip() + " "

            doc_tokens = preprocess_spacy(doc_string)
            doc_infos[docno] = {'tokens': doc_tokens, 'content': doc_string}

    return doc_infos

def add_inverted_index(tokens, docno):
  # data is the tokens of the docno document
  for word in tokens:
    if word not in inverted_index:
      inverted_index[word] = set() # a word can be in a doc multiple times, so no duplicates

    if word in inverted_index:
      inverted_index[word].add(docno)

count = 0
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)

    if os.path.isfile(filepath) and count == 0:
    #if os.path.isfile(filepath):
        print(filepath)
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
            doc_infos = process_document(content)

            # Update the inverted index
            for docno, info in doc_infos.items():
                tokens = info['tokens']
                content = info['content']
                #add docno
                document_ids.append(docno)

                for token in tokens:
                    if token not in inverted_index:
                      #using set
                        inverted_index[token] = {docno}
                    else:
                      #use sets to ensure there aren't any dups
                        inverted_index[token].add(docno)

        count += 1


#og VV
################################################################################

tfidf_vectorizer = TfidfVectorizer(tokenizer=preprocess_spacy)

def create_doc_matrix(rel_docs):
  tfidf_matrix = tfidf_vectorizer.fit_transform(rel_docs)
  return tfidf_matrix

def calc_cosine_sim(query, rel_docs):

    tfidf_matrix = create_doc_matrix(rel_docs)

    #transforming is necesary to ensure the query is using the same vectorizer
    query_tfidf = tfidf_vectorizer.transform([query])

    cs_list = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

    #sorts the indices based on similarity in descending order
    indices = np.argsort(cs_list)[::-1] # the minus 1 indicates that it goes backwards
    similarity_scores = cs_list[indices]

    return indices, similarity_scores

def retrieval_rank(query, inverted_index):
    query_tokens = preprocess_spacy(query) #want to tokenize using preprocess spacy

    # Find the right documents for the query using the inverted index
    # for each token in query, if the token is in the inverted index, the relevant document ids are added to the set
    relevant_documents = set()
    for token in query_tokens:
        if token in inverted_index:
            relevant_documents.update(inverted_index[token])

    relevant_documents = list(relevant_documents)

    rel_docs = []
    for doc in relevant_documents:
      rel_docs.append(document_strings[doc])

    # Compute cosine similarity scores for the query and all the documents
    # sorted_indices, sorted_similarities = compute_cosine_similarity(query, inverted_index, documents, relevant_documents) #here we are passing the relevant docs so we make sure we compute score fore the relevant ones
    indices, similarity_scores = calc_cosine_sim(query, rel_docs)

    rank = 1
    ranked_documents = []
    for i in range(0, len(rel_docs)):
      # add a tuple containing the document num and the cossim score
      ranked_documents.append((relevant_documents[indices[i]], rank, similarity_scores[i]))
      rank = rank + 1

    return relevant_documents, ranked_documents



# # make sure to change the set to list
# for token, docnos in inverted_index.items():
#     inverted_index[token] = list(docnos)

# # Example: Print the inverted index
# for token, docnos in inverted_index.items():
#     with open("result.txt", 'a') as file:
#         file.write("\n")
#         file.write(f"{token}: {docnos}")
#     #print(f"{token}: {docnos}")


def write_results(run_name, query_list, q_indices, num_of_res):
  f = open("Results.txt", "w")
  Q = "Q0"

  #loop through the queries
  for topic_id in q_indices:
    print(f"Retrieving and ranking documents for query: {topic_id+1}")
    rel_docs, rank_docs = retrieval_rank(query_list[topic_id], inverted_index)

    for docno, rank, score in rank_docs[:num_of_res]:
      print(f"{topic_id+1} {Q} {docno} {rank} {score} {run_name}\n")
      f.write(f"{topic_id+1} {Q} {docno} {rank} {score} {run_name}\n")

  f.close()

def create_queries(text_run):
  #text_run will be "title", or "title_desc"
  directory = 'CSI4107_A1/'

  f = os.path.join(directory, "topics1-50.txt")
  # Checking to see if 'f' is a valid file
  if os.path.isfile(f):
    print(f)
    textfile = open(f, 'r')
    filetext = textfile.read()
    textfile.close()

    #Replacing all /n with ' '
    file_string = filetext.replace("\n", " ")

    titles = re.findall("<title>(.*?)<desc>", file_string)
    print(len(titles))
    print(titles)

    #Checking to see if text_run is title_desc or title to determine if adding a description is necessary
    if(text_run == "title_desc"):
      descriptions = re.findall("<desc>(.*?)<narr>", file_string)
      print(len(descriptions))
      print(descriptions)

    queries = []
    #iterates over the titles extracted from a file, concatenating them with corresponding descriptions (if applicable) to create queries.
    for i in range(len(titles)):
      q = titles[i]
      if(text_run == "title_desc"):
        q = q + descriptions[i]
      queries.append(preprocess_spacy(q))
      #queries.append(q)

    #print(len(queries))
    print(queries)
    return queries

qlist_title_desc = create_queries("title_desc")
print("test1")
print(qlist_title_desc)
print("test2")
#write_results("run_q1_q25_title_desc",  qlist_title_desc, [0, 24], 10)



#Jessica vv
#################################################################################
# def calc_sim_score(query_tokens, rel_doc_tokens):
#   print("Calculating BM25 scores")
#   bm25 = BM25Okapi(rel_doc_tokens)

#   # get the similarity scores
#   doc_scores = bm25.get_scores(query_tokens)

#   # sort the list into decreasing order
#   indices = np.argsort(doc_scores)[::-1]
  
#   return doc_scores, indices

# def get_rel_docs(query_tokens):
#   print("Getting relevant documents")
#   rel_docno_set = set()
#   rel_docno_list = []
#   rel_doc_strings = []
#   rel_doc_tokens = []

#   # for every word in the query
#   for token in query_tokens:
#     if token in inverted_index:
#       # add the docs that contain the word to the set of rel docs
#       for docno in inverted_index[token]:
#         if docno not in rel_docno_set:
#           rel_docno_set.add(docno)
#           rel_docno_list.append(docno)
#           rel_doc_strings.append(document_strings[docno])
#           rel_doc_tokens.append(document_tokens[docno])

#   return rel_docno_set, rel_docno_list, rel_doc_tokens, rel_doc_strings

# def retrieval_rank(query):
#   # preprocess the query string into tokens
#   query_tokens = preprocess_spacy(query)

#   rd_set, rel_docno, rel_tokens, rel_doc_strings = get_rel_docs(query_tokens)

#   # Compute cosine similarity scores for the query and all the documents
#   # sorted_indices, sorted_similarities = compute_cosine_similarity(query, inverted_index, documents, relevant_documents) #here we are passing the relevant docs so we make sure we compute score fore the relevant ones
#   sim_scores, indices = calc_sim_score(query_tokens, rel_tokens)

#   rank = 1
#   ranked_documents = []
#   rank_docno = []
#   rank_docstrings = []
#   for i in range(0, len(rel_doc_strings)):
#     # add a tuple containing the document num and the cossim score
#     ranked_documents.append((rel_docno[indices[i]], rank, sim_scores[indices[i]]))
#     rank_docno.append(rel_docno[indices[i]])
#     rank_docstrings.append(rel_doc_strings[indices[i]])
#     rank = rank + 1

#   return rank_docstrings, ranked_documents, rank_docno

#################################################################################
