# README #
This program is related to the Touché task: https://webis.de/events/touche-21/shared-task-1.html.

Depending on the arguments, the program can index the Args.me corpus (https://zenodo.org/record/3734893#.YIV1SYNfi6k) and search for arguments given one or more text queries.

## Requirements
- The Args.me corpus: https://zenodo.org/record/3734893#.YIV1SYNfi6k
- Optionally, the topics of the Touché task: https://webis.de/events/touche-21/topics-task-1-only-titles-2021.zip

## Usage
### Build
Run `./mvnw clean package` to create a jar of the application. You will find the jar (with dependencies) in the `target` directory.

### Index
Run `java -jar target/indexing-1.0-SNAPSHOT-jar-with-dependencies.jar --index --dataset [path to the Args.me corpus] --output [path to the directory that will contain the index]`

### Search
Run `java -jar target/indexing-1.0-SNAPSHOT-jar-with-dependencies.jar --search --path {path to where the index is located} --queries {path to the file containing the queries} --results {path to where the retrieved arguments will be written} --max {maximum number of retrieved arguments} --titleboost {boost assigned to a match of a query with the title}`

Note that the queries file must have the following XML format:

```
<topics>
    <topic>
        <number>1</number>
        <title>Query text 1</title>
    </topic>
    <topic>
        <number>2</number>
        <title>Query text 2</title>
    </topic>
</topics>
```