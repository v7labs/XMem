build: 
	docker build . -t xmem-training  --progress plain 

train:
	docker run -it --rm --gpus all -v ./:/app --shm-size 20g xmem-training