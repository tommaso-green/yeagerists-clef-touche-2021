import os
import json
import xmltodict


# todo do query expansion - returns arrays of queries and ids
def expand_query(query: str):
    return [qid for qid in range(3)], [query + " 1" for i in range(3)]


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


def read_topics(topics_path):

    f = open(topics_path, "r")
    xml_data = f.read()
    my_dict = xmltodict.parse(xml_data)

    topic_dicts = my_dict['topics']['topic']
    t_list = [(int(x['number']), x['title']) for x in topic_dicts]
    return sorted(t_list, key=lambda x: x[0])


def main():
    # todo add program arguments
    index_path = "data/index"
    queries_path = "data/_tmp_queries.xml"
    results_path = "data/res.txt"
    topics_path = "datasets/touche2021topics/topics-task-1-only-titles.xml"
    max_docs = "10"
    topic_list = read_topics(topics_path)
    # TODO read queries from topics file
    ids, queries = expand_query("weed")
    #ids = [x[0] for x in topic_list]
    #queries = [x[1] for x in topic_list]
    write_queries_to_file(queries_path, queries, ids)

    args = "--search --path {index} --queries {queries} --results {results} --max {max_docs}" \
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
