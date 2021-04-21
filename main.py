import os
import json


# todo do query expansion - returns arrays of queries and ids
def expand_query(query: str):
    return [qid for qid in range(3)], [query+" 1" for i in range(3)]


def write_queries_to_file(path: str, queries: [str], ids: [str]):
    with open(path, "w+") as f:
        f.write("<topics>")

        for i in range(len(queries)):
            f.write("<topic>")

            f.write("<number>")
            f.write(str(ids[i]))
            f.write("</number>")

            f.write("<title>")
            f.write(queries[i])
            f.write("</title>")

            f.write("</topic>")

        f.write("</topics>")


def read_results(res_path: str):
    res = []
    with open(res_path) as f:
        res = json.load(f)
    return res


def main():

    # todo add program arguments
    index_path = "data/index"
    queries_path = "data/_tmp_queries.xml"
    results_path = "data/res.txt"
    max_docs = "10"

    # TODO read queries from topics file
    ids, queries = expand_query("weed")
    write_queries_to_file(queries_path, queries, ids)

    args = "--search --path {index} --queries {queries} --results {results} --max {max_docs}"\
        .format(index=index_path, queries=queries_path, results=results_path, max_docs=max_docs)
    os.system("java -jar indexing/target/indexing-1.0-SNAPSHOT-jar-with-dependencies.jar " + args)

    documents = read_results(results_path)
    for d in documents:
        print("\n\n")
        print("Query id: %s" % d["queryId"])
        print("Doc id: %s" % d["id"])
        print("Doc body: %s" % d["body"])
        print("Doc stance: %s" % d["stance"])
        print("Doc score: %s" % d["score"])

    # todo argument quality reranking etc


if __name__ == "__main__":
    main()
