python -m nmt.nmt \
--src=vi \
--tgt=en \
--hparams_path=/data/xueyou/github/nmt/nmt/standard_hparams/iwslt15_coverage.json \
--out_dir=/data/xueyou/data/iwslt15/normed_coverage_context_bahdanau_n \
--vocab_prefix=/data/xueyou/data/iwslt15/vocab \
--train_prefix=/data/xueyou/data/iwslt15/train \
--dev_prefix=/data/xueyou/data/iwslt15/tst2012 \
--test_prefix=/data/xueyou/data/iwslt15/tst2013
