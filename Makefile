include .env
export

build: build-jar install-requirements build-trec

index: build-jar index

build-jar:
	# Build Lucene-based jar
	cd indexing && ./mvnw.cmd clean package -DskipTests=true -e

build-trec:
	# Build trec to evaluate performance
	cd trec_eval-9.0.7 && make

install-requirements:
	# Install python requirements
	pip install -r requirements.txt

index:
	mkdir -p ${INDEX_PATH}
	java -jar indexing/target/indexing-1.0-SNAPSHOT-jar-with-dependencies.jar  --index -dataset ${DATASET_PATH} -output ${INDEX_PATH}

run:
	python main.py -m ${MAX_RESULTS} --ckpt ${MODEL} --topicpath ${TOPICS} --alpha ${ALPHA} --name ${RUN_NAME} --titleboost ${TITLE_BOOST} --type ${TYPE} --beta ${BETA} --queryexp ${QUERY_EXP}

evaluate:
	./trec_eval-9.0.7/trec_eval -m all_trec ${JUDGMENTS} ./data/experiment/${RUN_NAME}/run.txt




