# Make sure the SSH session has the correct version of Python on its path.
# You will probably have to change the line below.
export PATH=/home/ubuntu/anaconda3/envs/tensorflow_p36/bin/:$PATH
ray start --redis-address=172.31.29.235:6379
