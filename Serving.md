## 如何导出NMT模型做Serving

建议使用稳定版本的tensorflow比如1.4.1,以及稳定版本的nmt比如tf-1.4分支来训练和导出模型。具体流程如下：

- 先用nmt tf-1.4训练好模型
- 由于tf-1.4.1版本的tensorflow在导出模型时有bug，需要手动修改一下tensorflow的代码，参见这个[commit](https://github.com/tensorflow/tensorflow/commit/af8a5507937108a41781ba117fa16edd3b1091b5)
    - 这个bug参见[issue](https://github.com/tensorflow/tensorflow/issues/14143)，简单说就是当设置clear_device=True的时候，会将一些dataset的map function丢失，导致serving的时候报function not defined的错误。
- 由于nmt使用了tf.data模块，在每次进行inference的时候，都必须要先做initialize，目前的解决方案是导出initializer和generate为两个operation，并且利用tf.control_dependencies保证必须先初始化initializer。具体参见inference.py中的export\_model方法。
- 目前只是简单的利用inference的时候加载的inference model做为导出，所以代码会依赖inference的一些参数。这个后期可以修掉。
- 导出后就可以用了。使用的实例可以参见surreal/generate/jobs_desc_gen.py代码，会有两次predict操作。
- serving目前的最新版本是1.4，可以直接安装binary

