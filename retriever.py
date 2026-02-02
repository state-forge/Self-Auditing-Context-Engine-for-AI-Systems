import os
from datetime import datetime
from langchain_community.document_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import PyPDFParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

def ret():

    CUTOFF_DATE = datetime(2024, 1, 1).timestamp()

    # Embedding setup 
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    
    try:
        hf = HuggingFaceBgeEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        return None, None

    # Check if database exists
    db_exists = os.path.exists("faiss_index")

    if db_exists:
        update = input("Database found! Do you want to update it? (y/n): ")
    else:
        update = "y"

    if update.lower() == "y":
        filepath = input("What is the folder path: ")
        
        if not os.path.exists(filepath):
            print(f"Error: Path '{filepath}' does not exist!")
            return None, None

        try:
            # Document loaders
            loader = GenericLoader(
                blob_loader=FileSystemBlobLoader(
                    path=filepath,
                    glob="*.pdf",
                ),
                blob_parser=PyPDFParser(),
            )
            document = loader.load()
            
            if not document or len(document) == 0:
                print("No PDF files found in the specified path!")
                return None, None
                
        except Exception as e:
            print(f"Error loading documents: {e}")
            return None, None
    
        

        try:
            # Split in chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
            )
            final_chunks = text_splitter.split_documents(document)
            
        except Exception as e:
            print(f"Error splitting documents: {e}")
            return None, None

        try:
            # Create and Save
            print("Creating vector database...")
            db = FAISS.from_documents(final_chunks, hf)
            db.save_local("faiss_index")
            print("Database saved.")
            
        except Exception as e:
            print(f"Error creating/saving database: {e}")
            return None, None
        
    else:
        print("Loading existing database...")
        try:
            db = FAISS.load_local("faiss_index", hf, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Error loading database: {e}")
            return None, None

    # Search
    query = input("What is question you want to ask: ") 
    
    if not query.strip():
        print("Query cannot be empty!")
        return None, None
    
    try:
        results = db.similarity_search_with_relevance_scores(query)

        SCORE_THRESHOLD=0.7

        results.sort(key=lambda x: x[1], reverse=True)


        source_file=[]

        final_chunks_revised=[]
        
        for docs, score in results:

            source_path=docs.metadata.get('source')

            file_timestamp=os.path.getctime(source_path)
            
            check=0
            for file in source_file:
                if file==docs.metadata.get('source'):
                    check+=1
            
            


            if (check<1):

                if (file_timestamp > CUTOFF_DATE):

                    if (score>0.85):
                        print(f"{docs.metadata.get('source')} is highly relevant.")

                    elif ((score>0.7) and (score<0.85)):
                        print(f"{docs.metadata.get('source')} is relevant.")

                    else:
                        print(f"{docs.metadata.get('source')} is not relevant.")

                    final_chunks_revised.append((docs,score))
                    
                
                else:
                    print(f"{docs.metadata.get('source')} is Outdated.")


                source_file.append(docs.metadata.get('source'))
            
        final_chunks_revised_revised = []
        for doc,score in final_chunks_revised:
            if (score>=0.7):
                final_chunks_revised_revised.append(docs)
            if (score<0.7):
                break
        return query, final_chunks_revised_revised
        
    except Exception as e:
        print(f"Error during search: {e}")
        return None, None
