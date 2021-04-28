import os
import json
import subprocess
import sys

import torch.cuda
import xmltodict
from argument_quality.model import *
from query_expansion_python.query_exp_utils import *
from datetime import datetime
import math


def main(args=None):
    if not args:
        args = parse_args(sys.argv[1:])

    dir_path = f"./data/experiment/{args.name}"
    try:
        os.makedirs(dir_path)
    except OSError:
        print(f"Error creating results directory {dir_path}, if the dir already exists change run name")
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
    if args.queryexp:
        start = datetime.now()
        print('-->Starting Query Expansion')

        ids, new_queries = expand_queries(queries)
        write_queries_to_file(args.querypath, new_queries, ids)

        print('Time taken for Query Expansion: ', datetime.now() - start)
    else:
        print('\n--->No query expansion')
        write_queries_to_file(args.querypath, queries, ids)

    java_args = f"--search --path {args.indexpath} --queries {args.querypath} --results {args.resultpath} --max {args.maxdocs} --titleboost {args.titleboost}"
    os.system("java -jar indexing/target/indexing-1.0-SNAPSHOT-jar-with-dependencies.jar " + java_args)

    # Each result document contains: queryId, id, body, stance, score
    documents = read_results(args.resultpath)
    print(f"--->Number of documents retrieved from index: {len(documents)}")

    if args.alpha > 0:
        start = datetime.now()

        print(f'Running quality re-ranking for {len(documents)} arguments')
        # Add 'total_score' to each doc considering predicted quality
        documents = get_quality_score(arg_quality_model, documents, args.alpha)
        re_ranked_docs = sorted(documents, key=lambda doc: (int(doc["queryId"]), -doc["total_score"]))

        print('Time for quality re-ranking: ', datetime.now() - start)
    else:
        print("\n--> No Argument Quality Reranking")
        for i, d in enumerate(documents):
            d["total_score"] = d["score"]
        re_ranked_docs = sorted(documents, key=lambda doc: (int(doc["queryId"]), -doc["total_score"]))

    save_run(documents=re_ranked_docs, directory=dir_path, args=args)


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


def expand_query(query: str):
    new_queries_list = impr_generate_similar_queries(query, verbose=False)
    return [query_id for query_id in range(len(new_queries_list))], new_queries_list


def score(alpha, rel_score, q_score, type, **kwargs):
    if type == 'sigmoid':
        sigmoid = lambda x: 1 / (1 + math.exp(-kwargs['beta'] * x))
        return (1 - alpha) * sigmoid(rel_score) + alpha * sigmoid(q_score)
    if type == 'normalize':
        return (1 - alpha) * rel_score/kwargs['max_rel'] + alpha * q_score/kwargs['max_q']
    if type == 'hybrid':
        sigmoid = lambda x: 1 / (1 + math.exp(-kwargs['beta'] * x))
        return (1 - alpha) * rel_score / kwargs['max_rel'] + alpha * sigmoid(q_score)


def expand_queries(queries: [str]):
    ret_ids = []
    ret_queries = []
    for q in queries:
        print(f"Expanding query {q}")
        ids, new_queries = expand_query(q)
        ret_ids.append(ids)
        ret_queries.append(queries)

    return ret_ids, ret_queries


def get_quality_score(model, documents, alpha):
    arguments = [d["body"] for d in documents]
    args_with_score = []

    # todo check if there's any improvement by dividing arguments in small batches
    for arg in arguments:
        args_with_score += model(arg)

    max_rel = max([d["score"] for d in documents])
    max_q = max([args_with_score[i][1] for i in range(len(args_with_score))])
    print(f"MAX RELEVANCE = {max_rel} \n MAX QUALITY = {max_q}")
    type = 'normalize'
    for i, d in enumerate(documents):
            if type == 'sigmoid':
                d["total_score"] = score(alpha, d["score"], args_with_score[i][1], type='sigmoid', beta=0.2)
            if type == 'normalize':
                d["total_score"] = score(alpha, d["score"], args_with_score[i][1], type='normalize',
                                         max_rel=max_rel, max_q=max_q)
            if type == 'hybrid':
                d["total_score"] = score(alpha, d["score"], args_with_score[i][1], type='hybrid',
                                         max_rel=max_rel, beta=0.2)

    return documents


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
    parser.add_argument('-qe', '--queryexp', action='store_true')
    parser.add_argument('-a', '--alpha', type=float, default=0.3)
    parser.add_argument('-tb', '--titleboost', type=float, default=0)
    parser.add_argument('-n', '--name', type=str, default="dev_run")

    return parser.parse_args(args)


if __name__ == "__main__":
    start_m = datetime.now()
    main()
    end_m = datetime.now()
    time_taken = end_m - start_m
    print('Time: ', time_taken)
