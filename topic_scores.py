import os
import argparse

import numpy
import pandas
import subprocess
import csv
import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str)
    parser.add_argument('-r', '--run_csv', type=str)
    parser.add_argument('--judgements', type=str)
    args = parser.parse_args()

    df = pandas.read_csv(args.run_csv)
    run_ids = list(df.get("ID"))
    folder_ids = os.listdir(args.directory)
    runs = [run for run in folder_ids if run in run_ids]
    run_ndcg_list = []
    limit = 150
    for i, run in enumerate(runs[limit:]):
        run_dir = "/".join([args.directory, run])
        if os.path.exists(f"{run_dir}/ndcg_topic.txt"):
            os.remove(f"{run_dir}/ndcg_topic.txt")
            print("***Deleted previous file***")
        else:
            print("###The file does not exist. New one will be created###")
        subprocess.check_output(
            f"./trec_eval-9.0.7/trec_eval -q -m ndcg_cut {args.judgements} {run_dir}/run.txt >> {run_dir}/ndcg_topic.txt",
            shell=True)
        f = open(f"{run_dir}/ndcg_topic.txt")
        lines = f.readlines()
        f.close()
        ndcg_per_topic = {}
        ndcg_cut_5_lines = []
        ndcg_total = None
        ndcg_total_line = None
        for line in lines:
            tokens = line.split("\t")
            if tokens[0].strip() == "ndcg_cut_5":
                if tokens[1].strip() != "all":
                    ndcg_per_topic[int(tokens[1])] = float(tokens[2])
                    line = " ".join([t.strip() for t in tokens]) + "\n"
                    ndcg_cut_5_lines.append(line)
                else:
                    line = " ".join([t.strip() for t in tokens]) + "\n"
                    ndcg_total_line = line
                    ndcg_total = float(tokens[2])
        print(ndcg_per_topic)
        run_ndcg_list.append(ndcg_per_topic)
        f = open(f"{run_dir}/ndcg_cut_5.txt", "w")
        f.truncate(0)
        ndcg_cut_5_lines.sort(key=lambda x: int(x.split(" ")[1]))
        f.write(ndcg_total_line)
        f.writelines(ndcg_cut_5_lines)
        f.close()

        wandb.init(project="ArgumentRetrieval_Tests", entity="yeagerists", id=run, resume=True)
        wandb.save(f"{run_dir}/ndcg_cut_5.txt")
        wandb.log({"nDCG@5_corr": ndcg_total})

        data = [[label, val] for (label, val) in ndcg_per_topic.items()]
        data.sort(key=lambda x: x[0])
        table = wandb.Table(data=data, columns=["topic", "nDCG@5"])
        fields = {"value": "nDCG@5", "label": "topic", "title": "nDCG@5 per topic"}
        chart = wandb.plot_table(vega_spec_name="yeagerists/vertical_bar_chart", data_table=table, fields=fields)
        wandb.log({"nDCG@5 per Topic Table": chart})

        # data = [[x] for x in ndcg_per_topic.values()]
        # data.sort(key=lambda x: x[0])
        # table = wandb.Table(data=data, columns=["nDCG@5"])
        # wandb.log({'my_histogram': wandb.plot.histogram(table, "nDCG@5", title="Histogram of nDCG@5 per topic")})
        #
        # data = list(ndcg_per_topic.values())
        # hist = numpy.histogram(data)
        # wandb.log({"histo_test": wandb.Histogram(np_histogram=hist)})

        wandb.finish()
        print(f"Processed run {run} ({i+1}/{len(runs)})")

    # topic_list = sorted(run_ndcg_list[0].keys())
    # with open('run_csv/summary_nDCG_topic.csv', 'w') as csvfile:
    #     csvfile.write(",".join(runs) + "\n")
    #     for topic in topic_list:
    #         value_list = [str(x[topic]) for x in run_ndcg_list]
    #         line = ",".join(value_list) + "\n"
    #         csvfile.write(line)
    #     csvfile.close()
    # wandb.init(project="ArgumentRetrieval_Tests", entity="yeagerists", name="overall_results_csv")  # , id=run)
    # wandb.save(f"run_csv/summary_nDCG_topic.csv")
    # wandb.finish()



main()
