python -m nmt.nmt \
--src=vi \
--tgt=en \
--hparams_path=nmt/standard_hparams/iwslt15_scheduled_sampling.json \
--out_dir=/data/xueyou/data/iwslt15/iwslt15_scheduled_sampling \
--vocab_prefix=/data/xueyou/data/iwslt15/vocab \
--train_prefix=/data/xueyou/data/iwslt15/train \
--dev_prefix=/data/xueyou/data/iwslt15/tst2012 \
--test_prefix=/data/xueyou/data/iwslt15/tst2013