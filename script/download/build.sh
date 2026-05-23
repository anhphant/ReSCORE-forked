python -m source.run.preprocess_raw_data \
--dataset_name musique; \
python -m source.run.generate_passage_embeddings \
--model_name_or_path Lee1219/iqatr-musique \
--passages ./data/embed_ready_data/musique.tsv \
--output_dir ./data/database/contriever_msmarco/musique \
--shard_id 0 \
--num_shards 1 \
--per_gpu_batch_size 256; \
python -m source.run.build_index \
--output_dir ./data/database/contriever_msmarco/musique; \

