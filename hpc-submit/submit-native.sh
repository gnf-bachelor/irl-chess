#BSUB -J sunfish-native
#BSUB -o sunfish-native%J.out
#BSUB -e sunfish-native%J.err
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=32G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
# end of BSUB options

module load python3
source irl-chess-env/bin/activate
python3 -m project.sunfish_native_pw
