.PHONY: buildCPU buildGPU buildBaseCPU pushBaseCPU dockerCPU dockerRunCPU dockerGPU dockerRunGPU

buildCPU:
	mkdir -p build;
	cd build; cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU=OFF
	cd build; make -j

buildGPU:
	mkdir -p build;
	cd build; cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU=ON -DARCHITECTURE=61 -DTHRUST_INCLUDE_CUB_CMAKE=ON
	cd build; make -j

buildBaseCPU:
	docker build . -f Dockerfile.baseCPU -t khbence/covid_ppcu:base_cpu

pushBaseCPU: buildBaseCPU
	docker push khbence/covid_ppcu:base_cpu

dockerCPU:
	docker build . -f Dockerfile.CPU -t khbence/covid_ppcu:cpu

dockerRunCPU: dockerCPU
	docker run khbence/covid_ppcu:cpu

dockerGPU:
	docker build . -f Dockerfile.GPU -t khbence/covid_ppcu:gpu

dockerRunGPU: dockerGPU
	docker run khbence/covid_ppcu:gpu