import re
import os
from multiprocessing import Pool, cpu_count
from collections import Counter
from nltk.stem import PorterStemmer as NltkPorterStemmer
import nltk

nltk.download('punkt')

# Tokenization and Text Parsing Functions
def tokenize(text):
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())

def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return set(file.read().split())

def remove_stopwords(tokens, stopwords):
    return [token for token in tokens if token not in stopwords]

def process_file(args):
    filepath, stopwords = args
    doc_dict = {}
    term_counter = Counter()

    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
            documents = re.findall(r'<DOC>(.*?)</DOC>', content, re.DOTALL)
            
            for doc in documents:
                doc_id_match = re.search(r'<DOCNO>(.*?)</DOCNO>', doc)
                if doc_id_match:
                    doc_id = doc_id_match.group(1).strip()
                    doc_dict[doc_id] = 1  # Placeholder for document ID assignment
                
                    text_match = re.search(r'<TEXT>(.*?)</TEXT>', doc, re.DOTALL)
                    if text_match:
                        text = text_match.group(1).strip()
                        tokens = tokenize(text)
                        tokens = remove_stopwords(tokens, stopwords)
                        term_counter.update(tokens)
    
    except Exception as e:
        print(f"Error processing file {filepath}: {str(e)}")
    
    return doc_dict, term_counter

def parse_documents(folder_path, stopwords):
    print(f"Parsing documents in folder: {folder_path}")
    file_list = [(os.path.join(folder_path, filename), stopwords) 
                 for filename in os.listdir(folder_path) 
                 if os.path.isfile(os.path.join(folder_path, filename))]
    
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_file, file_list)
    
    doc_dict = {}
    term_counter = Counter()
    for doc_dict_part, term_counter_part in results:
        doc_dict.update(doc_dict_part)
        term_counter.update(term_counter_part)
    
    print(f"Total documents processed: {len(doc_dict)}")
    print(f"Total unique terms before stemming: {len(term_counter)}")
    
    # Sort tokens alphabetically and stem them using nltk's Porter Stemmer
    unique_words = sorted(term_counter.keys())
    stemmer = NltkPorterStemmer()
    stemmed_words = {word: stemmer.stem(word) for word in unique_words}
    
    # Create final term dictionary with stemmed words
    term_dict = {}
    for stemmed_term in sorted(set(stemmed_words.values())):
        term_dict[stemmed_term] = len(term_dict) + 1
    
    print(f"Total unique terms after stemming: {len(term_dict)}")
    
    # Assign proper IDs to documents
    for i, doc_id in enumerate(sorted(doc_dict.keys())):
        doc_dict[doc_id] = i + 1
    
    return term_dict, doc_dict

def write_output(term_dict, doc_dict):
    output_file_path = 'parser_output.txt'
    print(f"Writing output to {output_file_path}")
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write("Term Dictionary:\n")
        # Sort terms alphabetically
        for term, term_id in sorted(term_dict.items()):
            output_file.write(f"{term}\t{term_id}\n")
        
        output_file.write("\nDocument Dictionary:\n")
        # Sort document names alphabetically
        for doc_name, doc_id in sorted(doc_dict.items()):
            output_file.write(f"{doc_name}\t{doc_id}\n")
    
    print(f"Output written to {output_file_path}")

if __name__ == "__main__":
    stopwords_file = 'stopwordlist.txt'
    documents_folder = 'ft911'
    
    print("Starting document parsing process")
    print(f"Loading stopwords from {stopwords_file}")
    stopwords = load_stopwords(stopwords_file)
    print(f"Loaded {len(stopwords)} stopwords")
    
    term_dict, doc_dict = parse_documents(documents_folder, stopwords)
    write_output(term_dict, doc_dict)
    print("Document parsing process completed")
