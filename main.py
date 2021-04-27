import os
import json
import torch.cuda
import xmltodict
from argument_quality.model import *
from query_expansion_python.query_exp_utils import *
from datetime import datetime
import math


def expand_query(query: str):
    new_queries_list = impr_generate_similar_queries(query, verbose=False)
    return [query_id for query_id in range(len(new_queries_list))], new_queries_list


def score(alpha, rel_score, q_score, type):
    if type == 'sigmoid':
        c = 0.2
        sigmoid = lambda x: 1 / (1 + math.exp(-c * x))
        return (1 - alpha) * sigmoid(rel_score) + alpha * sigmoid(q_score)


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
    f.close()


def read_results(res_path: str):
    res = []
    with open(res_path) as f:
        res = json.load(f)
    f.close()
    return res


def read_topics(topics_path):
    f = open(topics_path, "r")
    xml_data = f.read()
    my_dict = xmltodict.parse(xml_data)

    topic_dicts = my_dict['topics']['topic']
    t_list = [(int(x['number']), x['title']) for x in topic_dicts]
    return sorted(t_list, key=lambda x: x[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--indexpath', type=str, default="data/index")
    parser.add_argument('-q', '--querypath', type=str, default="data/_tmp_queries.xml")
    parser.add_argument('-r', '--resultpath', type=str, default="data/res.txt")
    parser.add_argument('-t', '--topicpath', type=str,
                        default="datasets/touche2021topics/topics-task-1-only-titles.xml")
    parser.add_argument('-c', '--ckpt', type=str, default="argument_quality/model_checkpoints/bert-base-uncased_best-epoch=04-val_r2=0.69.ckpt")
    parser.add_argument('-m', '--maxdocs', type=str, default="10")
    parser.add_argument('-qe', '--queryexp', action='store_true')
    parser.add_argument('-a', '--alpha', type=float, default=0.3)
    parser.add_argument('-n', '--name', type=str, default="dev_run")
    args = parser.parse_args()

    topic_list = read_topics(args.topicpath)
    print(f"Topic List size: {len(topic_list)}")

    arg_quality_model = ArgQualityModel.load_from_checkpoint(args.ckpt)
    arg_quality_model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device {device}")
    arg_quality_model.to(device)

    ids = [x[0] for x in topic_list]
    queries = [x[1] for x in topic_list]
    if args.queryexp:
        start = datetime.now()
        print('-->Starting Query Expansion')
        tot_queries = []  # just for debugging
        for q in queries:
            print(f"Expanding query {q}")
            ids, new_queries = expand_query(q)
            tot_queries.append(new_queries)
            write_queries_to_file(args.querypath, new_queries, ids)
        end = datetime.now()
        time_taken = end - start
        print('Time taken for Query Expansion: ', time_taken)
    else:
        print('\n--->No query expansion')
        write_queries_to_file(args.querypath, queries, ids)
    java_args = f"--search --path {args.indexpath} --queries {args.querypath} --results {args.resultpath} --max {args.maxdocs}"
    os.system("java -jar indexing/target/indexing-1.0-SNAPSHOT-jar-with-dependencies.jar " + java_args)

    documents = read_results(args.resultpath)
    print(f"--->Documents retrieved {len(documents)}")
    for d in documents:
        print("\n\n")
        print("Query id: %s" % d["queryId"])
        print("Doc id: %s" % d["id"])
        print("Doc body: %s" % d["body"])
        print("Doc stance: %s" % d["stance"])
        print("Doc score: %s" % d["score"])

    if args.alpha > 0:
        arguments = [d["body"] for d in documents]
        args_with_score = []
        start = datetime.now()
        # todo check if there's any improvement by dividing arguments in small batches
        for arg in arguments:
            args_with_score += arg_quality_model(arg)
        end = datetime.now()
        time_taken = end - start
        print('Time for Query Expansion: ', time_taken)

        print("\n" + "*" * 5 + "RERANKED LIST" + "*" * 5)
        for i, d in enumerate(documents):
            d["total_score"] = score(args.alpha, d["score"], args_with_score[i][1], type='sigmoid')
        reranked_docs = sorted(documents, key=lambda d: (int(d["queryId"]), -d["total_score"]))
        for i, d in enumerate(reranked_docs):
            print("\n\n")
            print("Query id: %s" % d["queryId"])
            print("Doc id: %s" % d["id"])
            print("Doc body: %s" % d["body"])
            print("Doc stance: %s" % d["stance"])
            print("Doc score: %s" % d["score"])
            print("Doc quality: %s" % args_with_score[i][1])
            print("Doc total score: %s" % d["total_score"])

    else:
        print("\n--> No Argument Quality Reranking")
        for i, d in enumerate(documents):
            d["total_score"] = d["score"]
        reranked_docs = sorted(documents, key=lambda d: (int(d["queryId"]), -d["total_score"]))

    rank = 0
    query_counter = reranked_docs[0]['queryId']
    with open("run.txt", "w") as f:
        for d in reranked_docs:
            if d['queryId'] == query_counter:
                rank += 1
            else:
                rank = 1
                query_counter = d['queryId']
            f.write(f"{d['queryId']} QO {d['id']} {rank} {d['total_score']:.3f} {args.name}\n")

    f.close()


if __name__ == "__main__":
    start_m = datetime.now()
    main()
    end_m = datetime.now()
    time_taken = end_m - start_m
    print('Time: ', time_taken)
