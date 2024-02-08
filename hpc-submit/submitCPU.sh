#BSUB -J sunfish-permuted
#BSUB -o sunfish-permuted%J.out
#BSUB -e sunfish-permuted%J.err
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=16G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
# end of BSUB options

module load python3/3.10.13
source venv/bin/activate 
python3 -m project.sunfish_permuted