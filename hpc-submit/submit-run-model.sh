#BSUB -J sunfish-run
#BSUB -o /zhome/e0/2/169222/Desktop/irl-chess/hpc-logs/sunfish-run/%J.out
#BSUB -e /zhome/e0/2/169222/Desktop/irl-chess/hpc-logs/sunfish-run/%J.err
#BSUB -q hpc
#BSUB -n 24
#BSUB -R "rusage[mem=1G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
# end of BSUB options

module load python3
source irl-chess-env/bin/activate
python3 -m irl_chess.run_model
