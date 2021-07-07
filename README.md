# README #

## Requirements
- Make
- The Args.me corpus: https://zenodo.org/record/3734893#.YIV1SYNfi6k
- The Argument Quality models: https://github.com/tommaso-green/yeagerists-clef-touche-2021/releases/download/v1.0.0/quality_models.zip

## Usage

Copy the `.env.example` file content and create a `.env` file and fill the variables according to your settings.
The parameters that MUST be defined in the `.env` file are:
- The Args.me corpus location: variable `DATASET_PATH`
- The argument quality model to be used (if ALPHA>0): variable `MODEL`

To run the system on a custom set of queries, change the variable `TOPICS`, which must contain the path to a file containing queries with the same format of `touche/topics-task-1-2020.xml`.

For more information about the `.env` file variables, check the comments available in `.env.example`.

### Build
- (Optional) Run `virtualenv venv` (or `python3 -m venv venv`) to create a virtual environment for the requirements 
    - Run `. venv/bin/activate` to activate the virtual env
    - Once you are done, run `deactivate` to close the virtual env
    - Note: the program has been tested with python 3.7, use `virtualenv venv -p=/usr/bin/python3.7` to specify the python version
    - Note2: if you don't have venv installed, run `python3 -m pip install --user virtualenv`
- Run `make build`
  - This command will build the Lucene-based jar, install python requirements, and build trec-eval.
  
Optionally, you can build only the Lucene jar by running `make build-jar`.

### Index
After downloading the Args.me corpus and creating the `.env` file: run `make index`.

### Run
Run `make run` to search for the queries specified in `.env` (variable `TOPICS`) to get run.txt` file associating each query with a set of arguments.

If you are using query expansion (variable `QUERY_EXP` in `.env`), you need to download the related model (see requirements).

If you are using argument quality re-ranking (i.e. if variable `ALPHA` is >0 in `.env`), you need to download the related model (see requirements).

### Evaluate
If you have a .qrels judgments file (read from variable `JUDGMENTS` in `.env`), run `make evaluate` to evaluate the performance of the program. 

The command uses the `.env` variable `RUN_NAME` to pick the run-file contained in the folder data/experiment (i.e. the chosen run-file will be `data/experiment/${RUN_NAME}/run.txt`) and run trec-eval and output the score.  