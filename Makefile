
build: 
	docker build --tag master-thesis .
run:
	docker run -it \
		--rm \
		-v  /root/data:$$(PWD)\data \
		--name thesisC \
		master-thesis bash