# pip install paddlepaddle-gpu # train on GPU
pip install paddlepaddle
pip install numpy
pip install matplotlib

#curl "http://ai-atest.bj.bcebos.com/cifar-10-python.tar.gz" -O cifar-10-python.tar.gz
!wget "http://ai-atest.bj.bcebos.com/cifar-10-python.tar.gz" -O cifar-10-python.tar.gz

# move cifar-10-python.tar.gz ./dataset
!mv cifar-10-python.tar.gz  ./dataset/