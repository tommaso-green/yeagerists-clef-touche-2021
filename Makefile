include .env
export

index: build index

build:
	# Build Lucene-based jar
	cd indexing && ./mvnw clean package -DskipTests=true -e
	# Install python requirements
	pip install -r requirements.txt
	# Build trec to evaluate performance
	cd trec_eval-9.0.7 && make


index:
	mkdir ${INDEX_PATH}
	java -jar indexing/target/indexing-1.0-SNAPSHOT-jar-with-dependencies.jar  --index -dataset ${DATASET_PATH} -output ${INDEX_PATH}

run:
	python main.py -m ${MAX_RESULTS} --ckpt ${MODEL} --topicpath ${TOPICS} --alpha ${ALPHA}

evaluate:
	./trec_eval-9.0.7/trec_eval -m all_trec ${JUDGMENTS} run.txt




