## 运行

这里只是存放训练和导出的bash脚本，实际使用时，注意运行目录应该在nmt的根目录下。



## 结果

train\_headline\_gnmt
```
model path:
/data/xueyou/textsum/headline/headline_gnmt_4_layer

"best_bleu": 14.86522483249639
"best_rouge": 27.184768997224502
```


train\_desc\_char:
```
model path:
/data/xueyou/data/sku_desc/sku_desc_s2s_char

dev ppl 13.60, dev bleu 3.7, dev rouge 13.8, test ppl 12.48, test bleu 3.9, test rouge 13.9
```

## Finetune

关于finetune，目前我采用的方法是：
- 将模型copy到新的目录下，修改checkpoint文件指向新的路径
- 修改hparams的参数，将train,test和dev的文件目录修改，train_steps提高，vocab保持原来的文件目录
- 修改outputdir，best\_bleu和best\_rouge目录指向新路径
- 其他保持不变，然后重新训练
