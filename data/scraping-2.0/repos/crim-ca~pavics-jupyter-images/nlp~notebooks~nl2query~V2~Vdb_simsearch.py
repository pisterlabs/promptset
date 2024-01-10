import os

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


def generate_ngrams(text, max_words):
    words = text.split()
    output = [] 
    ngrams_dict = {}
    for x in range(1,max_words+1):
        for i in range(len(words)- x+1):
            ngram = " ".join(words[i:i+x])
            output.append(ngram)
            # add 1-grams
            ngrams_dict[ngram] = words[i:i+x]
            # add 2-grams in case of 3-gram and more
            if x >= 2:
                for j in range(0, len(ngrams_dict[ngram])-1):
                    ngrams_dict[ngram].append(" ".join(words[i+j:i+j+2]))
    return output, ngrams_dict


class Vdb_simsearch():
    """ class to handle vector database """
    
    def __init__(self, prop_vdb_path, prop_vocab_file, targ_vdb_path, targ_vocab_file) -> None:
        self.prop_vdb_path = prop_vdb_path
        self.prop_vocab_file = prop_vocab_file
        self.targ_vdb_path = targ_vdb_path
        self.targ_vocab_file = targ_vocab_file
        self.embeddings = HuggingFaceEmbeddings(
            model_name='intfloat/e5-base-v2',
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        
        # set up property vdb
        self.prop_csv_loader = CSVLoader(file_path=self.prop_vocab_file, csv_args={
        'delimiter': '#',
        'quotechar': '"',
        'fieldnames': ['propval', 'description']}, 
        source_column="propval")
        self.prop_db = self.get_vdb(self.prop_vdb_path, self.prop_csv_loader, self.text_splitter, self.embeddings)
        
        # set up target vdb
        self.targ_csv_loader = CSVLoader(file_path=self.targ_vocab_file, csv_args={
            'delimiter': '#',
            'quotechar': '"',
            'fieldnames': ['varname', 'aliases', 'description']}, 
            source_column="varname") #specify a source for the document created from each row. Otherwise file_path will be used as the source for all documents created from the CSV file.
        self.targ_db = self.get_vdb(self.targ_vdb_path, self.targ_csv_loader, self.text_splitter, self.embeddings)


    def get_vdb(self, db_dir, csv_loader, text_splitter, embeddings):
        """Create or read existing vdb from given directory"""
        if os.path.exists(db_dir):
            print("Loading Chroma Vdb from...", db_dir)
            return Chroma(persist_directory=db_dir, embedding_function=embeddings)
        else:
            print("Creating Chroma Vdb at...", db_dir)
            documents = csv_loader.load()
            texts = text_splitter.split_documents(documents)
            db = Chroma.from_documents(texts, embedding=embeddings, persist_directory=db_dir)
            # Save vector database as persistent files in the output folder
            db.persist()
            return db
        

    def query_one_target(self, query:str, k:int=15, score_t:float=0.72, verbose:bool=False):
        relevant = self.targ_db.similarity_search_with_relevance_scores(query, k=k, 
                                                                    include_metadata=True, 
                                                                    score_threshold=score_t)
        rel_docs = []
        scores = []
        if verbose:
            print("\nQUERY: ", query)
            print("RESULTS: ", len(relevant))
        for (t,score) in relevant:
            rel = ""
            result = t.page_content.split("\n")
            v = result[0]
            a = result[1]
            if v.startswith("varname: "):
                v = v[9:]
                rel += (v)
            if len(a)>9 and a.startswith("aliases: "):
                a = a[9:]
                rel += (", "+a)
            if verbose:
                print(rel,score)
            rel_docs.append(rel)
            scores.append(score)
        return rel_docs, scores


    def query_ngram_target(self, query:str, ngrams:int=3, threshold:float=0.72, verbose:bool=False):
        # generate ngrams up to length 3 by default
        ngrams_list, ngrams_dict = generate_ngrams(query, ngrams) 
        ngrams_list += [query]
        ngram_results = {}
        ngram_scores = {}
        for ngrams in ngrams_list:
            # remember which results come from which query to identify span
            ngram_results[ngrams], ngram_scores[ngrams] = self.query_one_target(ngrams, score_t=threshold, verbose=verbose)
                
        # join ngram results
        if verbose:
            print("\nJOINT RESULTS:")
        join_results = {}
        join_scores = {}
        top_score = 0
        top_span = ""
        max_len = 0
        max_span = ""
        
        for k,v in ngram_results.items():
            if verbose:
                print("")
                print(k, len(v))
            join_results[k] = v
            join_scores[k] = ngram_scores[k]
            if k!= query and " " in k: # not full query nor 1-gram
                for ngram in ngrams_dict[k]:
                    if len(ngram_results[ngram]) > 0:
                        add_list = [r for r in ngram_results[ngram] if r not in join_results[k]]
                        join_results[k] += add_list
                        index_scores = [ngram_results[ngram].index(e) for e in add_list]
                        join_scores[k] += [ngram_scores[ngram][s] for s in index_scores]
                    if verbose:
                        print(ngram, len(ngram_results[ngram]))
            if len(join_scores[k])>0:
                # highest average score results
                join_avg = sum(join_scores[k])/len(join_scores[k])
                if verbose:
                    print("AVG :", join_avg)
                if join_avg > top_score:
                    top_score = join_avg
                    top_span = k
            if len(join_results[k]) > max_len:
                # highest length results
                max_len = len(join_results[k])
                max_span = k
            if verbose:
                print("LEN :", len(join_results[k]))
        if verbose:
            print("\nBEST RESULT:")
            print(max_len, max_span, join_results[max_span])
        if max_span:
            # take 
            res = join_results[max_span][:20]
            # return top results above a threshold
            return max_span, res
        else:
            return "", ""


    def query_one_prop(self, query, k=5, score_t=0.72, verbose=False):
        if verbose:
            print("\nQUERY: ", query)
        relevant = self.prop_db.similarity_search_with_relevance_scores(query, k=k, 
                                                                    include_metadata=True, 
                                                                    score_threshold=score_t)
        rel_docs = []
        for (t,score) in relevant:
            v = t.page_content.split("\n")[0]
            if v.startswith("propval: "):
                v = v[9:]
            if verbose:
                print(v, score)
            rel_docs.append((v, score))
        return rel_docs


    def query_ngram_prop(self, query, ngrams=3, threshold=0.6, verbose=False):
        collect_results = []
        # generate ngrams up to length 3
        ngrams_list, _ = generate_ngrams(query, ngrams)
        ngrams_list += [query]
        ngram_results = {}
        for ngrams in ngrams_list:
            rel_docs = self.query_one_prop(ngrams, score_t=threshold, verbose=verbose)
            # remember which results come from wihch query to identify span
            if len(rel_docs) > 0:
                ngram_results[ngrams] = rel_docs
                collect_results+=(rel_docs)
            else:
                ngram_results[ngrams] = []
                
        # sort results by descending score
        collect_results = sorted(collect_results, key= lambda x: x[1], reverse=True)
        
        top_res = {}
        if collect_results:
            # remove duplicates
            set_results = [collect_results[0]]
            for (t,score) in collect_results:
                if t not in list(zip(*set_results))[0]:
                    set_results.append((t,score))
                    
            # return highest score results
            for item in set_results:
                # find span (ngram) that resulted this
                for k,v in ngram_results.items():
                    if item in v:
                        if k not in top_res.keys():
                            # take only highest score first item per span
                            top_res[k]= item
        if verbose:
            print("\nTOP RESULTS:", top_res)
        # return top result
        top_res = list(top_res.items())
        if len(top_res) > 0:
            # TODO above a threshold or other condition
            return top_res[0]
        else:
            return "", ""


if __name__ == "__main__":
    my_vdbs = Vdb_simsearch(
        "nl2query/V2/prop_vdb",
        "nl2query/V2/prop_vocab.csv",
        "nl2query/V2/target_vdb",
        "nl2query/V2/target_vocab3.csv",
    )
    
    query = "sentinel daily rain amount"
    my_vdbs.query_ngram_target(query, verbose=True)
