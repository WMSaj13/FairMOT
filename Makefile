all: build download-model test

build:
	# build base image with dependencies
	docker build -t fairmot-base --target fairmot-base .

	# compile cuda dependent code with image and download model
	docker run --shm-size=1g --ulimit memlock=-1 \
	    --ulimit stack=67108864 --gpus all --rm -it \
	    -v ${PWD}:/workspace/FairMOT \
			-w /workspace/FairMOT/src/lib/models/networks/DCNv2 \
			fairmot-base bash make.sh

	# build image with cuda dependent module
	docker build -t fairmot --target fairmot .

download-model:
	docker run --rm -it \
		-v ${PWD}:/workspace/FairMOT \
		-w /workspace/FairMOT/models \
		fairmot \
		gdown 'https://drive.google.com/uc?id=1udpOPum8fJdoEQm6n0jsIgMMViOMFinu'


test:
	docker run --shm-size=1g --ulimit memlock=-1 \
	    --ulimit stack=67108864 --gpus all --rm -it \
	    -v ${PWD}:/workspace/FairMOT \
			-w /workspace/FairMOT/src \
	    fairmot \
			python demo.py mot \
				--load_model ../models/all_dla34.pth \
				--conf_thres 0.4

# custom video
test2:
	docker run --shm-size=1g --ulimit memlock=-1 \
	    --ulimit stack=67108864 --gpus all --rm -it \
	    -v ${PWD}:/workspace/FairMOT \
			-w /workspace/FairMOT/src \
	    fairmot \
			python demo.py mot \
				--input-video ../videos/gold.mp4 \
				--output-root ../results2 \
				--load_model ../models/all_dla34.pth \
				--conf_thres 0.4
