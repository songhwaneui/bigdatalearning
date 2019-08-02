"M"
input_dataset = tf.data.TextLineDataset(os.path.join(self.data_dir, "train.inputs"))
batched_input_dataset = input_dataset.batch(self.hparams.batch_size)
input_iterator = batched_input_dataset.make_initializable_iterator()
batch_input = input_iterator.get_next()
batch_input.set_shape([self.hparams.batch_size])
words = tf.string_split(batch_input, " ")
word_ids = word2id.lookup(words)
dense_word_ids = tf.sparse_tensor_to_dense(word_ids)
# shape = [batch_size, time]

line_number = word_ids.indices[:, 0]
line_position = word_ids.indices[:, 1]
lengths = tf.segment_max(data=line_position,
                         segment_ids=line_number) + 1




"Q"
self._logger.info("End of epoch %d." % (epochs_completed+1))
save_path = saver.save(sess, "saves/model.ckpt", global_step=global_step_val)
self._logger.info("Model saved at: %s" % save_path)



"I"
dense_word_ids = tf.constant(word_ids)
lengths = tf.constant(len(word_ids))
# Insert batch dimension.
dense_word_ids = tf.expand_dims(dense_word_ids, axis=0)
lengths = tf.expand_dims(lengths, axis=0)

with tf.variable_scope("inference", reuse=False):
    logits = self._inference(dense_word_ids, lengths)
predictions = tf.argmax(logits, axis=1)



"E"
with tf.variable_scope("bi-RNN"):
    # Build RNN layers
    rnn_cell_forward = tf.contrib.rnn.GRUCell(self.hparams.rnn_hidden_dim)
    rnn_cell_backward = tf.contrib.rnn.GRUCell(self.hparams.rnn_hidden_dim)

    # Apply dropout to RNN
    if self.hparams.dropout_keep_prob < 1.0:
        rnn_cell_forward = tf.contrib.rnn.DropoutWrapper(rnn_cell_forward, output_keep_prob=self._dropout_keep_prob_ph)
        rnn_cell_backward = tf.contrib.rnn.DropoutWrapper(rnn_cell_backward, output_keep_prob=self._dropout_keep_prob_ph)

    # Stack multiple layers of RNN
    rnn_cell_forward = tf.contrib.rnn.MultiRNNCell([rnn_cell_forward] * self.hparams.rnn_depth)
    rnn_cell_backward = tf.contrib.rnn.MultiRNNCell([rnn_cell_backward] * self.hparams.rnn_depth)

    (output_forward, output_backward), _ = tf.nn.bidirectional_dynamic_rnn(
        rnn_cell_forward, rnn_cell_backward,
        inputs=layer_out,
        sequence_length=lengths,
        dtype=tf.float32
    )
    hiddens = tf.concat([output_forward, output_backward], axis=-1)
    # shape = [batch_size, time, rnn_dim*2]




"C"
# Number of possible output categories.
output_dim = len(self.id2label)
vocab_size = len(self.id2word) + 1
embeddings = tf.get_variable(
    "embeddings",
    shape=[vocab_size, self.hparams.embedding_dim],
    initializer=tf.initializers.variance_scaling(
        scale=1.0, mode="fan_out", distribution="uniform")
)
embedded = tf.nn.embedding_lookup(embeddings, inputs)
# shape = [batch_size, time, embed_dim]
layer_out = embedded



"A"

with open(os.path.join(data_dir, "train.vocab"), "r") as _f_handle:
    vocab = [l.strip() for l in list(_f_handle) if len(l.strip()) > 0]
if len(vocab) > hparams.vocab_size:
    vocab = vocab[:hparams.vocab_size]

self.id2word = vocab
self.word2id = {}
for i, word in enumerate(vocab):
    self.word2id[word] = i




"J"
saver = tf.train.Saver()
saver.restore(sess, saved_file)
pred_val = sess.run(
    [predictions],
    feed_dict={self._dropout_keep_prob_ph: 1.0}
)[0]



"O"
loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                         name="cross_entropy")
loss_op = tf.reduce_mean(loss_op, name='cross_entropy_mean')
train_op = tf.train.AdamOptimizer().minimize(loss_op, global_step=global_step)

eval = tf.nn.in_top_k(logits, labels, 1)
correct_count = tf.reduce_sum(tf.cast(eval, tf.int32))
accuracy = tf.divide(correct_count, tf.shape(labels)[0])




"P"
try:
    accuracy_val, label_ids_val, loss_val, global_step_val, _ = sess.run(
        [accuracy, labels, loss_op, global_step, train_op],
        feed_dict={self._dropout_keep_prob_ph: self.hparams.dropout_keep_prob}
    )
    if global_step_val % 10 == 0:
        self._logger.info("[Step %d] loss: %.4f, accuracy: %.2f%%" %(global_step_val, loss_val, accuracy_val * 100))
except tf.errors.OutOfRangeError:
    # End of epoch.
    break




"B"
with open(os.path.join(data_dir, "label.vocab"), "r") as _f_handle:
    labels = [l.strip() for l in list(_f_handle) if len(l.strip()) > 0]
labels.insert(0, "PAD")
self.id2label = labels
self.label2id = {}
for i, label in enumerate(labels):
    self.label2id[label] = i



"L"

word2id = tf.contrib.lookup.index_table_from_tensor(
    mapping=tf.constant(self.id2word),
    num_oov_buckets=1,
    name="word2id"
)

label2id = tf.contrib.lookup.index_table_from_tensor(
    mapping=tf.constant(self.id2label),
    default_value=self.label2id["O"],
    name="label2id"
)



"G"

with tf.variable_scope("read-out"):
    prev_layer_size = layer_out.get_shape().as_list()[1]
    weight = tf.get_variable("weight", shape=[prev_layer_size, output_dim],
                             initializer=tf.initializers.variance_scaling(
                                 scale=2.0, mode="fan_in", distribution="normal"
                             ))
    bias = tf.get_variable("bias", shape=[output_dim],
                           initializer=tf.initializers.zeros())
    predictions = tf.add(tf.matmul(layer_out, weight), bias, name='predictions')



"D"

layer_out = tf.layers.dense(
    inputs=layer_out,
    units=self.hparams.rnn_hidden_dim,
    activation=tf.nn.relu,
    kernel_initializer=tf.initializers.variance_scaling(
        scale=1.0, mode="fan_avg", distribution="normal"),
    name="input_projection"
)




"F"
mask = tf.sequence_mask(lengths)
bi_lstm_out = tf.reshape(tf.boolean_mask(hiddens, mask), [-1, self.hparams.rnn_hidden_dim * 2])
layer_out = bi_lstm_out  # shape=[sum of seq length, 2*LSTM hidden layer size]




"N"
label_dataset = tf.data.TextLineDataset(os.path.join(self.data_dir, "train.labels"))
batched_label_dataset = label_dataset.batch(self.hparams.batch_size)
label_iterator = batched_label_dataset.make_initializable_iterator()
batch_label_str = label_iterator.get_next()
batch_label = tf.string_split(batch_label_str, " ")
label_ids = label2id.lookup(batch_label)
dense_label_ids = tf.sparse_tensor_to_dense(label_ids)
# shape = [batch_size, time]



"K"

pred_str = [self.id2label[i] for i in pred_val]
for word, tag in zip(sentence, pred_str):
    print("%s[%s]" %(word, tag), end=' ')




"H"
sentence = word_tokenize(sentence)
word_ids = []
for word in sentence:
    if word in self.word2id:
        word_ids.append(self.word2id[word])
    else:
        word_ids.append(len(self.word2id))
