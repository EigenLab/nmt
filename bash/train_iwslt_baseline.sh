python -m nmt.nmt \
--src=vi \
--tgt=en \
--hparams_path=nmt/standard_hparams/iwslt15.json \
--out_dir=/home/xueyou/train/iwslt15_baseline \
--vocab_prefix=/data/xueyou/data/iwslt15/vocab \
--train_prefix=/data/xueyou/data/iwslt15/train \
--dev_prefix=/data/xueyou/data/iwslt15/tst2012 \
--test_prefix=/data/xueyou/data/iwslt15/tst2013