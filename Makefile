DATA_PATH="$(pwd)/data"

build: 
	docker build --tag master-thesis .

run:
	docker run -it --rm \
		-p 8888:8888 \
		master-thesis \
		-v "$(pwd)"/notebooks:/home/neuro/notebooks \
		-v $(DATA_PATH):/ADNI \
		jupyter notebook