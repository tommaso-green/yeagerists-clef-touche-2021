# README #

## Requirements
- Make
- The Args.me corpus: https://zenodo.org/record/3734893#.YIV1SYNfi6k
- Optionally, the topics of the Touché task: https://webis.de/events/touche-21/topics-task-1-only-titles-2021.zip
- TODO add other requirements

## Usage

Copy the `.env.example` file content and create a `.env` file with the following variables:
- DATASET_PATH=/path/to/args.me/corpus
- INDEX_PATH=./path/to/where/the/index/will/be/stored/or/read
- MAX_RESULTS=maximum number of documents retrieved by the searcher


### Build
Run `make build`.

### Index
After downloading the Args.me corpus and creating the `.env` file: run `make index`.

### todo

