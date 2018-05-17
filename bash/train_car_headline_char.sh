python -m nmt.nmt  \
   --src=source --tgt=target  \
   --hparams_path=nmt/standard_hparams/car_headline_char.json \
   --out_dir=/data/xueyou/car/car_slot_data/char/s2s_char  \
   --vocab_prefix=/data/xueyou/car/car_slot_data/char/vocab \
   --train_prefix=/data/xueyou/car/car_slot_data/char/train  \
   --dev_prefix=/data/xueyou/car/car_slot_data/char/dev \
   --test_prefix=/data/xueyou/car/car_slot_data/char/test