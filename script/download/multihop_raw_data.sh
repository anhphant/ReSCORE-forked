# This script is based on code from the IRCOT project by Stony Brook NLP.
# Source: https://github.com/StonyBrookNLP/ircot

# If gdown doesn't work, you can download files from mentioned URLs manually
# and put them at appropriate locations.
pip install gdown

mkdir -p .temp/s
mkdir -p data/raw_data

echo "\n\nDownloading raw musique data\n"
mkdir -p data/raw_data/musique
# URL: https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view?usp=sharing
gdown "1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h&confirm=t" -O .temp/musique_v1.0.zip
unzip -jo .temp/musique_v1.0.zip -d data/raw_data/musique -x "*.DS_Store"

rm -rf .temp/

# The resulting raw_data/ directory should look like:
# ── 2wikimultihopqa
# │   ├── dev.json
# │   ├── id_aliases.json
# │   ├── test.json
# │   └── train.json
# ├── hotpotqa
# │   ├── dev_random_20_single_hop_annotations.txt
# │   ├── wikpedia-paragraphs/
# │   ├──  ├── ...
# │   ├── hotpot_dev_distractor_v1.json
# │   └── train_random_20_single_hop_annotations.txt
# ├── iirc
# │   ├── context_articles.json
# │   ├── dev.json
# │   └── train.json
# └── musique
#     ├── dev_test_singlehop_questions_v1.0.json
#     ├── musique_ans_v1.0_dev.jsonl
#     ├── musique_ans_v1.0_test.jsonl
#     ├── musique_ans_v1.0_train.jsonl
#     ├── musique_full_v1.0_dev.jsonl
#     ├── musique_full_v1.0_test.jsonl
#     └── musique_full_v1.0_train.jsonl
