# Installing Caffe Cuda and all its dependencies on Virtual Box in Ubuntu 14.04 to make my first deep learning network!

1. sudo apt-get update && sudo apt-get upgrade
2. Download Vagrant from their website
3. Download Virtual Box from either website, or from Apps+update/upgrade. 
4. Install Ubuntu on the Virtual Box
5. Activate bidirectional clipboard, and shared folders. Start Virtual Box.
6. Install build essentials
	sudo apt-get install build-essential
7. Install latest version of kernel headers:
	sudo apt-get install linux-headers-`uname -r`
8. Install VBox Additions:
	Download VBOXADDITIONS (VBox guest additions)
	Mount it as a drive
	Run it
	Restart system
9. Install CUDA:
  sudo apt-get install curl
	Download appropriate CUDA s/w from Nvidia's website (the .run file)
	Run it under cd ~/Downloads/ as mentioned 
	Make the downloaded installer file runnable:
		chmod +x cuda_version_runfile.run (change name to name of your runfile)
	Accept the EULA
	Do NOT install the graphics card drivers (since we are in a virtual machine)
	Install the toolkit (leave path at default)
	Install symbolic link
	Install samples (leave path at default)
10. Update the library path:
	echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
	echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/lib' >> ~/.bashrc
	source ~/.bashrc
11. Install dependencies:
	sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev protobuf-compiler gfortran libjpeg62 libfreeimage-dev libatlas-base-dev git python-dev python-pip libgoogle-glog-dev libbz2-dev libxml2-dev libxslt-dev libffi-dev libssl-dev libgflags-dev liblmdb-dev python-yaml
	sudo easy_install pillow
12. Download Caffe:
	cd ~
	git clone https://github.com/BVLC/caffe.git
13. Install python dependencies for Caffe:
	cd caffe
	Install all the packages required stated under python/requirements.txt
14. Add a couple of symbolic links for some reason:
	sudo ln -s /usr/include/python2.7/ /usr/local/include/python2.7
	sudo ln -s /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ /usr/local/include/python2.7/numpy
15. Create a Makefile.config from the example:
	cp Makefile.config.example Makefile.config
	nano Makefile.config
	Uncomment the line # CPU_ONLY := 1 (In a virtual machine we do not have access to the the GPU)
	Under PYTHON_INCLUDE, replace /usr/lib/python2.7/dist-packages/numpy/core/include with /usr/local/lib/python2.7/dist-packages/numpy/core/include (i.e. add /local)
16. Compile Caffe:
	make pycaffe
	make all
	make test
17. Download the ImageNet Caffe model and labels:
	./scripts/download_model_binary.py models/bvlc_reference_caffenet
	./data/ilsvrc12/get_ilsvrc_aux.sh
