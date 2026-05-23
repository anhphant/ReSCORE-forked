# This script is based on code from the IRCOT project by Stony Brook NLP.
# Source: https://github.com/StonyBrookNLP/ircot

# mkdir -p source/metrics


git clone https://github.com/stonybrooknlp/musique source/evaluation/official_evaluation/musique
cd source/evaluation/official_evaluation/musique ; git checkout 24cc5b297acc2abfc5fb3d0becb6ef7b73d03717
cd ../../../../
