from retriever import ret
from answer_generator import ans_gen

def main():
    query, results = ret()
    if query is None or results is None:
        return
    else:
        ans_gen(query, results)
        return
if __name__ == "__main__":
    main()