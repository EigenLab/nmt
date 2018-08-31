import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _compute_attention
from tensorflow.python.framework import ops, tensor_shape
from tensorflow.python.util import nest
from tensorflow.python.ops.rnn_cell_impl import _linear,_zero_state_tensors
import collections
from tensorflow.python.ops import math_ops

def _coverage_bahdanau_score(processed_query, keys, coverage, normalize):
    """Implements Bahdanau-style (additive) scoring function.
    Args:
        processed_query: Tensor, shape `[batch_size, num_units]` to compare to keys.
        keys: Processed memory, shape `[batch_size, max_time, num_units]`.
        coverage: [batch_size, max_time, num_units]
    Returns:
        A `[batch_size, max_time]` tensor of unnormalized score values.
    """
    dtype = processed_query.dtype
    # Get the number of hidden units from the trailing dimension of keys
    num_units = keys.shape[2].value or tf.shape(keys)[2]
    # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
    processed_query = tf.expand_dims(processed_query, 1)
    v = tf.get_variable(
        "attention_v", [num_units], dtype=dtype)
    if normalize:
        # Scalar used in weight normalization
        g = tf.get_variable(
            "attention_g", [1], dtype=dtype,
            initializer= lambda shape,dtype,partition_info: (tf.sqrt((1. / num_units))))
        # Bias added prior to the nonlinearity
        b = tf.get_variable(
            "attention_b", [num_units], dtype=dtype,
            initializer=tf.zeros_initializer())
        # normed_v = g * v / ||v||
        normed_v = g * v * tf.rsqrt(
            tf.reduce_sum(tf.square(v)))
        return tf.reduce_sum(
            normed_v * tf.tanh(keys + processed_query + coverage + b), [2])
    else:
        return tf.reduce_sum(v * tf.tanh(keys + processed_query + coverage), [2])


class CoverageBahdanauAttention(tf.contrib.seq2seq.BahdanauAttention):
    def __init__(self,
                num_units,
                memory,
                memory_sequence_length=None,
                normalize=False,
                probability_fn=None,
                score_mask_value=float("-inf"),
                name="CoverageBahdanauAttention"):
        """Construct the Attention mechanism.
        Args:
            num_units: The depth of the query mechanism.
            memory: The memory to query; usually the output of an RNN encoder.  This
            tensor should be shaped `[batch_size, max_time, ...]`.
            memory_sequence_length (optional): Sequence lengths for the batch entries
            in memory.  If provided, the memory tensor rows are masked with zeros
            for values past the respective sequence lengths.
            normalize: Python boolean.  Whether to normalize the energy term.
            probability_fn: (optional) A `callable`.  Converts the score to
            probabilities.  The default is @{tf.nn.softmax}. Other options include
            @{tf.contrib.seq2seq.hardmax} and @{tf.contrib.sparsemax.sparsemax}.
            Its signature should be: `probabilities = probability_fn(score)`.
            score_mask_value: (optional): The mask value for score before passing into
            `probability_fn`. The default is -inf. Only used if
            `memory_sequence_length` is not None.
            name: Name to use when creating ops.
        """
        super(CoverageBahdanauAttention, self).__init__(
                    num_units = num_units,
                    memory = memory,
                    memory_sequence_length = memory_sequence_length,
                    normalize = normalize,
                    probability_fn = probability_fn,
                    score_mask_value = score_mask_value,
                    name = name)

    def __call__(self, query, previous_alignments):
        """Score the query based on the keys and values.
        Args:
            query: Tensor of dtype matching `self.values` and shape
            `[batch_size, query_depth]`.
            previous_alignments: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]`
            (`alignments_size` is memory's `max_time`).
        Returns:
            alignments: Tensor of dtype matching `self.values` and shape
            `[batch_size, alignments_size]` (`alignments_size` is memory's
            `max_time`).
        """
        with tf.variable_scope(None, "pointer_generator_bahdanau_attention", [query]):
            processed_query = self.query_layer(query) if self.query_layer else query
            score = _coverage_bahdanau_score(processed_query, self._keys, previous_alignments, self._normalize)
        # Note: previous_alignments is not used in probability_fn in Bahda attention, so I use it as coverage vector in coverage mode
        alignments = self._probability_fn(score, previous_alignments)
        
        return alignments


class CoverageAttentionWrapperState(
    collections.namedtuple("CoverageAttentionWrapperState",
                           ("cell_state", "attention", "time", "coverages","alignments",
                            "alignment_history"))):
  """`namedtuple` storing the state of a `AttentionWrapper`.

  Contains:

    - `cell_state`: The state of the wrapped `RNNCell` at the previous time
      step.
    - `attention`: The attention emitted at the previous time step.
    - `time`: int32 scalar containing the current time step.
    - `alignments`: A single or tuple of `Tensor`(s) containing the alignments
       emitted at the previous time step for each attention mechanism.
    - `alignment_history`: (if enabled) a single or tuple of `TensorArray`(s)
       containing alignment matrices from all time steps for each attention
       mechanism. Call `stack()` on each to convert to a `Tensor`.
  """

  def clone(self, **kwargs):
    """Clone this object, overriding components provided by kwargs.

    Example:

    ```python
    initial_state = attention_wrapper.zero_state(dtype=..., batch_size=...)
    initial_state = initial_state.clone(cell_state=encoder_state)
    ```

    Args:
      **kwargs: Any properties of the state object to replace in the returned
        `AttentionWrapperState`.

    Returns:
      A new `AttentionWrapperState` whose properties are the same as
      this one, except any overridden properties as provided in `kwargs`.
    """
    return super(CoverageAttentionWrapperState, self)._replace(**kwargs)

class CoverageAttentionWrapper(tf.contrib.seq2seq.AttentionWrapper):
    def __init__(self,cell,
        attention_mechanism,
        attention_layer_size=None,
        alignment_history=False,
        cell_input_fn=None,
        output_attention=False,
        initial_cell_state=None,
        name=None):
        #assert isinstance(attention_mechanism, CoverageBahdanauAttention), "%r is not CoverageBahdanauAttention" % attention_mechanism
        super(CoverageAttentionWrapper, self).__init__(
            cell, 
            attention_mechanism, 
            attention_layer_size, 
            alignment_history,
            cell_input_fn,
            output_attention,
            initial_cell_state,
            name)
        self.coverage_layer = tf.layers.Dense(attention_mechanism._num_units,name='coverage_layer',use_bias=False)
        self.fertility_layer = tf.layers.Dense(1,name='fertility_layer',use_bias=False)

    @property
    def state_size(self):
        """The `state_size` property of `AttentionWrapper`.

        Returns:
        An `AttentionWrapperState` tuple containing shapes used by this object.
        """
        return CoverageAttentionWrapperState(
            cell_state=self._cell.state_size,
            time=tensor_shape.TensorShape([]),
            attention=self._attention_layer_size,
            coverages=self._item_or_tuple(a.alignments_size for a in self._attention_mechanisms),
            alignments=self._item_or_tuple(
                a.alignments_size for a in self._attention_mechanisms),
            alignment_history=self._item_or_tuple(
                () for _ in self._attention_mechanisms))  # sometimes a TensorArray

    def zero_state(self, batch_size, dtype):
        """Return an initial (zero) state tuple for this `AttentionWrapper`.
        **NOTE** Please see the initializer documentation for details of how
        to call `zero_state` if using an `AttentionWrapper` with a
        `BeamSearchDecoder`.
        Args:
        batch_size: `0D` integer tensor: the batch size.
        dtype: The internal state data type.
        Returns:
        An `AttentionWrapperState` tuple containing zeroed out tensors and,
        possibly, empty `TensorArray` objects.
        Raises:
        ValueError: (or, possibly at runtime, InvalidArgument), if
            `batch_size` does not match the output size of the encoder passed
            to the wrapper object at initialization time.
        """
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.zero_state(batch_size, dtype)
            error_message = (
                "When calling zero_state of AttentionWrapper %s: " % self._base_name +
                "Non-matching batch sizes between the memory "
                "(encoder output) and the requested batch size.  Are you using "
                "the BeamSearchDecoder?  If so, make sure your encoder output has "
                "been tiled to beam_width via tf.contrib.seq2seq.tile_batch, and "
                "the batch_size= argument passed to zero_state is "
                "batch_size * beam_width.")
            with tf.control_dependencies(
                self._batch_size_checks(batch_size, error_message)):
                cell_state = nest.map_structure(
                    lambda s: tf.identity(s, name="checked_cell_state"),
                    cell_state)
            return CoverageAttentionWrapperState(
                cell_state=cell_state,
                time=tf.zeros([], dtype=tf.int32),
                attention=_zero_state_tensors(self._attention_layer_size, batch_size,
                                                dtype),
                coverages=self._item_or_tuple(
                    _zero_state_tensors(attention_mechanism.alignments_size,batch_size,dtype)
                    for attention_mechanism in self._attention_mechanisms),
                alignments=self._item_or_tuple(
                    attention_mechanism.initial_alignments(batch_size, dtype)
                    for attention_mechanism in self._attention_mechanisms),
                # since we need to read the alignment history several times, so we need set clear_after_read to False
                alignment_history=self._item_or_tuple(
                    tf.TensorArray(dtype=dtype, size=0, clear_after_read=False,dynamic_size=True)
                    if self._alignment_history else ()
                    for _ in self._attention_mechanisms))
                  
    def call(self, inputs, state):
        """Perform a step of attention-wrapped RNN.
        - Step 1: Mix the `inputs` and previous step's `attention` output via
            `cell_input_fn`.
        - Step 2: Call the wrapped `cell` with this input and its previous state.
        - Step 3: Score the cell's output with `attention_mechanism`.
        - Step 4: Calculate the alignments by passing the score through the
            `normalizer`.
        - Step 5: Calculate the context vector as the inner product between the
            alignments and the attention_mechanism's values (memory).
        - Step 6: Calculate the attention output by concatenating the cell output
            and context through the attention layer (a linear layer with
            `attention_layer_size` outputs).
        Args:
            inputs: (Possibly nested tuple of) Tensor, the input at this time step.
            state: An instance of `AttentionWrapperState` containing
            tensors from the previous time step.
        Returns:
            A tuple `(attention_or_cell_output, next_state)`, where:
            - `attention_or_cell_output` depending on `output_attention`.
            - `next_state` is an instance of `AttentionWrapperState`
                containing the state calculated at this time step.
        Raises:
            TypeError: If `state` is not an instance of `AttentionWrapperState`.
        """
        if not isinstance(state,CoverageAttentionWrapperState):
            raise TypeError("Expected state to be instance of AttentionWrapperState. "
                            "Received type %s instead."  % type(state))

        # Step 1: Calculate the true inputs to the cell based on the
        # previous attention value.
        cell_inputs = self._cell_input_fn(inputs, state.attention)
        cell_state = state.cell_state
        
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

        cell_batch_size = (
            cell_output.shape[0].value or tf.shape(cell_output)[0])
        error_message = (
            "When applying AttentionWrapper %s: " % self.name +
            "Non-matching batch sizes between the memory "
            "(encoder output) and the query (decoder output).  Are you using "
            "the BeamSearchDecoder?  You may need to tile your memory input via "
            "the tf.contrib.seq2seq.tile_batch function with argument "
            "multiple=beam_width.")
        with tf.control_dependencies(
            self._batch_size_checks(cell_batch_size, error_message)):
            cell_output = tf.identity(
                cell_output, name="checked_cell_output")

        if self._is_multi:
            previous_alignments = state.alignments
            previous_alignment_history = state.alignment_history
            previous_coverage = state.coverages
        else:
            previous_alignments = [state.alignments]
            previous_alignment_history = [state.alignment_history]
            previous_coverage = [state.coverages]

        all_alignments = []
        all_attentions = []
        all_histories = []
        all_coverages = []

        n = tf.get_variable("N",[1],dtype=tf.float32,initializer=tf.constant_initializer(2))
        for i, attention_mechanism in enumerate(self._attention_mechanisms):
            # values on attention_mechanism is hidden state from encoder
            # batch * atten_len * 1
            fertility = n * tf.nn.sigmoid(self.fertility_layer(attention_mechanism.values))
            #fertility = self.N * tf.nn.sigmoid(self.fertility_layer(attention_mechanism.values))
            # coverage shape: batch * atten_len * 1
            expand_coverage = tf.expand_dims(previous_coverage[i],axis=-1)
            pre_coverage = self.coverage_layer(expand_coverage / fertility)
            #pre_coverage = tf.Print(pre_coverage,[fertility[0],previous_coverage[i][0],tf.reduce_sum(previous_coverage[i][0])],message='pre_coverage')

            attention, alignments = _compute_attention(
                attention_mechanism, cell_output, pre_coverage,
                self._attention_layers[i] if self._attention_layers else None)
            alignment_history = previous_alignment_history[i].write(
                state.time, alignments) if self._alignment_history else ()

            # batch * atten_len * 1
            coverage = expand_coverage + tf.expand_dims(alignments,axis=-1)

            all_alignments.append(alignments)
            all_histories.append(alignment_history)
            all_attentions.append(attention)
            all_coverages.append(tf.squeeze(coverage,axis=[2]))

        attention = tf.concat(all_attentions, 1)
        next_state = CoverageAttentionWrapperState(
            time=state.time + 1,
            cell_state=next_cell_state,
            attention=attention,
            coverages=self._item_or_tuple(all_coverages),
            alignments=self._item_or_tuple(all_alignments),
            alignment_history=self._item_or_tuple(all_histories))

        if self._output_attention:
            return attention, next_state
        else:
            return cell_output, next_state


class ContextLSTMCell(tf.contrib.rnn.BasicLSTMCell):
    def __init__(self, num_units, forget_bias=1.0,
               state_is_tuple=True, activation=None, reuse=None):
        super(ContextLSTMCell,self).__init__(num_units,forget_bias,state_is_tuple,activation,reuse)
        self.source_layer = tf.layers.Dense(4*self._num_units,use_bias=False, name='source_layer')
        self.target_layer = tf.layers.Dense(4*self._num_units,use_bias=False, name='target_layer')
        self.context_layer = tf.layers.Dense(self._num_units,use_bias=False, name='context_layer')


    def call(self, inputs, state):
        """Long short-term memory cell (LSTM).
        Args:
        inputs: `2-D` tensor with shape `[batch_size x input_size]`.
        state: An `LSTMStateTuple` of state tensors, each shaped
            `[batch_size x self.state_size]`, if `state_is_tuple` has been set to
            `True`.  Otherwise, a `Tensor` shaped
            `[batch_size x 2 * self.state_size]`.
        Returns:
        A pair containing the new hidden state, and the new state (either a
            `LSTMStateTuple` or a concatenated state, depending on
            `state_is_tuple`).
        """
        sigmoid = tf.nn.sigmoid
        # Parameters of gates are concatenated into one multiply for efficiency.
        c, h = state

        inputs, attention = tf.split(inputs, num_or_size_splits=2,axis=-1)
        z = tf.nn.sigmoid(self.context_layer(tf.concat([inputs,attention,h],axis=-1)))
        source_inputs = self.source_layer(tf.concat([inputs,h],axis=-1))
        target_inputs = self.target_layer(attention)
        tiled_z = tf.tile(z,[1,4])
        context = tiled_z * source_inputs + (1 - tiled_z) * target_inputs
        bias = tf.get_variable("context_bias",[4*self._num_units],dtype=tf.float32)

        context = tf.nn.bias_add(context, bias)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = tf.split(
            value=context, num_or_size_splits=4, axis=1)

        new_c = (
            c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
        new_h = self._activation(new_c) * sigmoid(o)

        new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
        return new_h, new_state