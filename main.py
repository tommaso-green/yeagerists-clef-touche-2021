import os
import json
import subprocess
import sys

import torch.cuda
import xmltodict
from argument_quality.model import *
from query_expansion_python.query_expansion_utils import *
from datetime import datetime
import math


def main(args=None):
    if not args:
        args = parse_args(sys.argv[1:])

    dir_path = f"./data/experiment/{args.name}"
    try:

        os.makedirs(dir_path)
    except OSError:
        print(f"Error creating results directory {dir_path}: directory already exists")
        if len(os.listdir(dir_path)) != 0:
            print("Directory already contains data: delete it or change run name")
            return

    topic_list = read_topics(args.topicpath)
    print(f"Topic List size: {len(topic_list)}")

    if args.alpha != 0:
        arg_quality_model = ArgQualityModel.load_from_checkpoint(args.ckpt)
        arg_quality_model.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device {device}")
        arg_quality_model.to(device)

    ids = [x[0] for x in topic_list]
    queries = [x[1] for x in topic_list]
    if args.queryexp == "yes":
        start = datetime.now()
        print('-->Starting Query Expansion')

        new_queries_lists = expand_queries(queries)

        # create a flat list of queries
        new_queries = []
        new_ids = []
        for i in range(len(new_queries_lists)):
            for q in new_queries_lists[i]:
                new_queries.append(q)
                new_ids.append(ids[i])  # replicate the original id for each expanded query
        queries = new_queries
        ids = new_ids
        write_queries_to_file(args.querypath, queries, ids)

        print('Time taken for Query Expansion: ', datetime.now() - start)
    else:
        print('\n--->No query expansion')
        write_queries_to_file(args.querypath, queries, ids)

    java_args = f"--search --path {args.indexpath} --queries {args.querypath} --results {args.resultpath} --max {args.maxdocs} --titleboost {args.titleboost}"
    os.system("java -jar indexing/target/indexing-1.0-SNAPSHOT-jar-with-dependencies.jar " + java_args)

    # Each result document contains: queryId, id, body, stance, score
    documents = read_results(args.resultpath)
    print(f"--->Number of documents retrieved from index: {len(documents)}")
    documents = remove_duplicate_documents(documents)
    print(f"--->Number of documents retrieved without duplicates: {len(documents)}")

    if args.alpha > 0:
        start = datetime.now()

        print(f'Running quality re-ranking')
        # Add 'total_score' to the top arg.nrerank documents for each topic

        # Group by query ID
        docs_per_query_id = dict()
        # Group documents by query id
        for doc in documents:
            docs_of_query = docs_per_query_id.get(doc["queryId"])
            if not docs_of_query:
                docs_of_query = []
            docs_of_query.append(doc)
            docs_per_query_id[doc["queryId"]] = docs_of_query

        re_ranked_docs = []
        for qid in docs_per_query_id:
            docs_per_query_id[qid].sort(key=lambda doc: doc["score"], reverse=True)
            top_reranked = get_quality_score(arg_quality_model, docs_per_query_id[qid][:args.nrerank], args)
            re_ranked_docs += top_reranked

        re_ranked_docs.sort(key=lambda doc: (int(doc["queryId"]), -doc["total_score"]))

        print('Time for quality re-ranking: ', datetime.now() - start)
    else:
        print("\n--> No Argument Quality Reranking")
        for i, d in enumerate(documents):
            d["total_score"] = d["score"]
        re_ranked_docs = sorted(documents, key=lambda doc: (int(doc["queryId"]), -doc["total_score"]))

    save_run(documents=re_ranked_docs, directory=dir_path, args=args)


def remove_duplicate_documents(documents):
    docs_per_query_id = dict()
    # Group documents by query id
    for doc in documents:
        docs_of_query = docs_per_query_id.get(doc["queryId"])
        if not docs_of_query:
            docs_of_query = []
        docs_of_query.append(doc)
        docs_per_query_id[doc["queryId"]] = docs_of_query
    # Remove duplicates
    res = []
    for qid, docs in docs_per_query_id.items():
        filtered_docs = []
        ids_set = set()
        for document in docs:
            if document["id"] not in ids_set:
                ids_set.add(document["id"])
                filtered_docs.append(document)
        res.extend(filtered_docs)
        print(f"Query {qid} has {len(filtered_docs)} documents")
    return res


def save_run(documents, directory, args):
    # Save runfile
    rank = 0
    # Get lowest query id (should be 1)
    query_counter = documents[0]['queryId']
    with open(directory + "/run.txt", "w") as f:
        for d in documents:
            if d['queryId'] == query_counter:
                rank += 1
            else:
                rank = 1
                query_counter = d['queryId']
            f.write(f"{d['queryId']} QO {d['id']} {rank} {d['total_score']:.3f} {args.name}\n")

    # Save arguments
    with open(directory + "/args.txt", "w") as f:
        f.write(str(args.__dict__))

    # Save nDCG@5 for quick read
    nDCG = float(subprocess.check_output("make -s evaluate | grep ndcg_cut_5 | head -1", shell=True).split()[2])

    with open(directory + "/ndcg5_" + str(nDCG), "w") as f:
        f.write(f"nDGC at 5: {nDCG}")


def score(alpha, rel_score, q_score, type, **kwargs):
    if type == 'sigmoid':
        sigmoid = lambda x: 1 / (1 + math.exp(-kwargs['beta'] * x))
        return (1 - alpha) * sigmoid(rel_score) + alpha * sigmoid(q_score)
    if type == 'normalize':
        return (1 - alpha) * rel_score / kwargs['max_rel'] + alpha * q_score / kwargs['max_q']
    if type == 'hybrid':
        sigmoid = lambda x: 1 / (1 + math.exp(-kwargs['beta'] * x))
        return (1 - alpha) * rel_score / kwargs['max_rel'] + alpha * sigmoid(q_score)


def expand_queries(queries: [str]):
    new_queries_list = expand_queries_list(queries, max_n_query=10, verbose=False)
    return new_queries_list


def get_quality_score(model, documents, args):
    # todo check if there's any improvement by dividing arguments in small batches
    for d in documents:
        d['quality'] = model(d['body'])[0][1]

    if args.type in ['normalize', 'hybrid']:
        max_rel = {}
        max_q = {}
        query_ids = set([d['queryId'] for d in documents])
        for query in query_ids:
            max_rel[query] = max([d['score'] for d in documents if d['queryId'] == query])
            max_q[query] = max([d['quality'] for d in documents if d['queryId'] == query])
        print(f"MAX RELEVANCE = {max_rel} \n MAX QUALITY = {max_q}")

    for d in documents:
        if args.type == 'sigmoid':
            d["total_score"] = score(args.alpha, d["score"], d["quality"], type='sigmoid', beta=args.beta)
        if args.type == 'normalize':
            d["total_score"] = score(args.alpha, d["score"], d["quality"], type='normalize',
                                     max_rel=max_rel[d["queryId"]], max_q=max_q[d["queryId"]])
        if args.type == 'hybrid':
            d["total_score"] = score(args.alpha, d["score"], d["quality"], type='hybrid',
                                     max_rel=max_rel[d["queryId"]], beta=args.beta)

    return documents


def write_queries_to_file(path: str, queries: [str], ids: [str]):
    assert len(queries) == len(ids)
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


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--indexpath', type=str, default="data/index")
    parser.add_argument('-q', '--querypath', type=str, default="data/_tmp_queries.xml")
    parser.add_argument('-r', '--resultpath', type=str, default="data/res.txt")
    parser.add_argument('-t', '--topicpath', type=str,
                        default="datasets/touche2021topics/topics-task-1-only-titles.xml")
    parser.add_argument('-c', '--ckpt', type=str,
                        default="argument_quality/model_checkpoints/bert-base-uncased_best-epoch=04-val_r2=0.69.ckpt")
    parser.add_argument('-m', '--maxdocs', type=str, default="10")
    parser.add_argument('-qe', '--queryexp', type=str, default="yes")
    parser.add_argument('-a', '--alpha', type=float, default=0.3)
    parser.add_argument('-tb', '--titleboost', type=float, default=0)
    parser.add_argument('-n', '--name', type=str, default="dev_run")
    parser.add_argument('--type', type=str, default="normalize")
    parser.add_argument('--beta', type=float, default=0.2)
    parser.add_argument('--nrerank', type=int, default=5)

    return parser.parse_args(args)


if __name__ == "__main__":
    start_m = datetime.now()
    main()
    end_m = datetime.now()
    time_taken = end_m - start_m
    print('Time: ', time_taken)
