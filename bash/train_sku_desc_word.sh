python -m nmt.nmt     --src=source --tgt=target     --hparams_path=nmt/standard_hparams/sku_desc_word.json     --out_dir=/data/xueyou/data/sku_desc/sku_desc_s2s_word_gru     --vocab_prefix=/data/xueyou/data/sku_desc/word_level/vocab     --train_prefix=/data/xueyou/data/sku_desc/word_level/train     --dev_prefix=/data/xueyou/data/sku_desc/word_level/dev     --test_prefix=/data/xueyou/data/sku_desc/word_level/test     --num_gpus=2