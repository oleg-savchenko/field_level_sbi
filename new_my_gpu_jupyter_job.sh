#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j-%N_slurm.out
## Activate right env
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load IPython/8.28.0-GCCcore-13.3.0
unset PYTHONPATH
unset CONDA_PREFIX
export PATH=$HOME/.local/bin:$PATH
which jupyter-notebook
# module load OpenMPI/4.1.4-GCC-11.3.0
# module load FFTW.MPI/3.3.10-gompi-2022a
# module load GSL/2.7-GCC-11.3.0
# module load HDF5/1.12.2-gompi-2022a
# module load jupyter-resource-usage/0.6.3-GCCcore-11.3.0
# module load jupyter-server/1.21.0-GCCcore-11.3.0
# module load jupyter-server-proxy/3.2.2-GCCcore-11.3.0
# module load UCX-CUDA/1.12.1-GCCcore-11.3.0-CUDA-11.7.0
# module load UCX/1.12.1-GCCcore-11.3.0
## source /home/osavchenko/venvs/tmnre/bin/activate
export OMP_NUM_THREADS=18


PORT=`shuf -i 5000-5999 -n 1`
PORT=8897
LOGIN_HOST=${SLURM_SUBMIT_HOST}-pub.snellius.surf.nl
BATCH_HOST=$(hostname)
# ssh -N -f -R ${PORT}:localhost:${PORT} ${BATCH_HOST}
echo "To connect to the notebook type the following command from your local terminal:"
echo "ssh -J ${USER}@${LOGIN_HOST} ${USER}@${BATCH_HOST} -L ${PORT}:localhost:${PORT}"
echo
echo "After connection is established in your local browser go to the address:"
echo "http://localhost:${PORT}"
jupyter notebook --no-browser --port $PORT --NotebookApp.allow_origin='*' --NotebookApp.ip='0.0.0.0'
