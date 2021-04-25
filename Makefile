include .env.example
export

index: build index

build:
	cd indexing && ./mvnw clean package -DskipTests=true -e

index:
	mkdir ${INDEX_PATH}
	java -jar indexing/target/indexing-1.0-SNAPSHOT-jar-with-dependencies.jar  --index -dataset ${DATASET_PATH} -output ${INDEX_PATH}
