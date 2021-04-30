# README #

## Requirements
- Make
- The Args.me corpus: https://zenodo.org/record/3734893#.YIV1SYNfi6k
- The Argument Quality models: https://drive.google.com/drive/folders/1x_oMuYorWAQ3JGBSCJKmFKewP_3YfuXW

## Usage

Copy the `.env.example` file content and create a `.env` file and fill the variables according to your settings.

### Build
- (Optional) Run `virtualenv venv` (or `python3 -m venv venv`) to create a virtual environment for the requirements 
    - Run `. venv/bin/activate` to activate the virtual env
    - Once you are done, run `deactivate` to close the virtual env
    - Note: the program was tested with python 3.7, use `virtualenv venv -p=/usr/bin/python3.7` to specify the python version
    - Note2: if you don't have venv installed, run `python3 -m pip install --user virtualenv`
- Run `make build`
  - This command will build the Lucene-based jar and install python requirements

### Index
After downloading the Args.me corpus and creating the `.env` file: run `make index`.

### Run
Run `make run` to search for the queries specified in `.env` (variable `TOPICS`) to get run.txt` file associating each query with a set of arguments.

### Evaluate
If you have a .qrels judgments file (read from variable `JUDGMENTS` in `.env`), run `make evaluate` to evaluate the performance of the program. 
