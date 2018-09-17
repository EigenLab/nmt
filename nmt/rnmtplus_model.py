# ======================================== 
# Author: Xueyou Luo 
# Email: xueyou.luo@aidigger.com 
# Copyright: Eigen Tech @ 2018 
# ========================================
"""RNMT+ attention sequence-to-sequence model with dynamic RNN support."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# TODO(rzhao): Use tf.contrib.framework.nest once 1.3 is out.
from tensorflow.python.util import nest

from . import attention_model
from . import gnmt_model
from . import model_helper
from .utils import misc_utils as utils
from .utils.attention_utils import CoverageAttentionWrapper,CoverageAttentionWrapperState

__all__ = ["RNMTPlusModel"]


class RNMTPlusModel(gnmt_model.GNMTModel):
  """Sequence-to-sequence dynamic model with RNMT+ attention architecture.
  """

  def __init__(self,
               hparams,
               mode,
               iterator,
               source_vocab_table,
               target_vocab_table,
               reverse_target_vocab_table=None,
               scope=None,
               extra_args=None):
    super(RNMTPlusModel, self).__init__(
        hparams=hparams,
        mode=mode,
        iterator=iterator,
        source_vocab_table=source_vocab_table,
        target_vocab_table=target_vocab_table,
        reverse_target_vocab_table=reverse_target_vocab_table,
        scope=scope,
        extra_args=extra_args)

  def _build_encoder(self, hparams):
    """Build a RNMT+ encoder."""
    if hparams.encoder_type == "uni" or hparams.encoder_type == "bi":
      return super(RNMTPlusModel, self)._build_encoder(hparams)

    if hparams.encoder_type != "rnmt+":
      raise ValueError("Unknown encoder_type %s" % hparams.encoder_type)

    # Build RNMT+ encoder.
    num_bi_layers = self.num_encoder_layers
    utils.print_out("  num_bi_layers = %d" % num_bi_layers)

    iterator = self.iterator
    source = iterator.source
    if self.time_major:
      source = tf.transpose(source)

    with tf.variable_scope("encoder") as scope:
      # Look up embedding, emp_inp: [max_time, batch_size, num_units]
      #   when time_major = True
      encoder_emb_inp = tf.nn.embedding_lookup(self.embedding_encoder,
                                               source)
    
      inputs = encoder_emb_inp
      encoder_states = []
      # Execute _build_bidirectional_rnn from Model class
      for i in range(num_bi_layers):
          with tf.variable_scope("layer_%d"%i) as scope:
            unit_type = 'weight_drop_lstm' if hparams.weight_drop else hparams.unit_type
            fw_cell = model_helper.create_rnn_cell(
                unit_type=unit_type,
                num_units=hparams.num_units // 2,
                num_layers=1,
                num_residual_layers=0,
                forget_bias=hparams.forget_bias,
                dropout=hparams.dropout,
                num_gpus=self.num_gpus,
                base_gpu=i,
                mode=self.mode,
                single_cell_fn=self.single_cell_fn)
            bw_cell = model_helper.create_rnn_cell(
                unit_type=unit_type,
                num_units=hparams.num_units // 2,
                num_layers=1,
                num_residual_layers=0,
                forget_bias=hparams.forget_bias,
                dropout=hparams.dropout,
                num_gpus=self.num_gpus,
                base_gpu=i,
                mode=self.mode,
                single_cell_fn=self.single_cell_fn)
            
            encoder_outputs_tup, encoder_state_tup =  tf.nn.bidirectional_dynamic_rnn(
                    fw_cell,
                    bw_cell,
                    inputs,
                    sequence_length=iterator.source_sequence_length,
                    dtype=inputs.dtype,
                    time_major=self.time_major)
            
            outputs = tf.concat(encoder_outputs_tup,axis=-1)
            encoder_state = model_helper.concat_reduce(encoder_state_tup[0],encoder_state_tup[1])
            encoder_states.append(encoder_state)
            if self.mode == tf.estimator.ModeKeys.TRAIN:
                 outputs = tf.nn.dropout(outputs, keep_prob=1-hparams.dropout)
            inputs = inputs + outputs

      # encoder_outputs: size [max_time, batch_size, num_units]
      #   when time_major = True
      # Pass all encoder state except the first bi-directional layer's state to
      # decoder.
    return inputs, tuple(encoder_states)