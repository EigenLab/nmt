python -m nmt.nmt \
--src=vi \
--tgt=en \
--debug=False \
--hparams_path=nmt/standard_hparams/iwslt15.json \
--out_dir=/data/xueyou/data/iwslt15/test_ngram_beam \
--vocab_prefix=/data/xueyou/data/iwslt15/vocab \
--train_prefix=/data/xueyou/data/iwslt15/train \
--dev_prefix=/data/xueyou/data/iwslt15/tst2012 \
--test_prefix=/data/xueyou/data/iwslt15/tst2013