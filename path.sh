# Theano 0.7.0 requires:
#   numpy 1.10.1 (1.8 failed)
#   scipy 0.16.1 (0.13 failed)
export THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32

export PYTHONPATH=$(pwd):$PYTHONPATH
# export PATH=/data/sls/scratch/dharwath/cuda-7.5/bin:$PATH
# export LIBRARY_PATH=/data/sls/scratch/dharwath/cuda-7.5/lib64:$LIBRARY_PATH
# export LD_LIBRARY_PATH=/data/sls/scratch/dharwath/cuda-7.5/lib64:$LD_LIBRARY_PATH
