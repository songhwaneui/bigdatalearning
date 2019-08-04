import argparse
import json
import collections
from datetime import datetime
import os
import logging
import math
import tensorflow as tf
from nltk.tokenize import word_tokenize
from data_process import CoNLL2003Processor

class SequenceTagger:
    def __init__(self, hparams, data_dir):
        self.hparams = hparams
        self.data_dir = data_dir
        self._dropout_keep_prob_ph = tf.placeholder(tf.float32, shape=[], name="dropout_keep_prob")
        self._logger = logging.getLogger(__name__)

        self.iterator_initializers = []

        """
        [A]
        Vocabulary(단어집) 파일을 로드합니다.
        단어 -> id, id -> 단어 변환 테이블을 생성합니다.

        """

        with open(os.path.join(data_dir, "train.vocab"), "r") as _f_handle:
            vocab = [l.strip() for l in list(_f_handle) if len(l.strip()) > 0]
        if len(vocab) > hparams.vocab_size:
            vocab = vocab[:hparams.vocab_size]

        self.id2word = vocab
        self.word2id = {}
        for i, word in enumerate(vocab):
            self.word2id[word] = i
        self.word2id["<UNK>"] = len(self.word2id)
        """
        [B]
        Label(태그 모음) 파일을 로드합니다.
        태그 -> id, id -> 태그 변환 테이블을 생성합니다.

        """
        with open(os.path.join(data_dir, "label.vocab"), "r") as _f_handle:
            labels = [l.strip() for l in list(_f_handle) if len(l.strip()) > 0]
        labels.insert(0, "PAD")
        self.id2label = labels
        self.label2id = {}
        for i, label in enumerate(labels):
            self.label2id[label] = i

        """
        [A'] 
        데이터 파일을 읽고 inputs, label example을 만드는 과정입니다.
        """

        self.conll_proc = CoNLL2003Processor(self.hparams, self.word2id, self.label2id)

    def _make_placeholders(self):
        # placeholders
        self.inputs_ph = tf.placeholder(shape=[None, None], dtype=tf.int32, name="inputs_ph")
        self.lengths_ph = tf.placeholder(shape=[None], dtype=tf.int32, name="lengths_ph")

        # 둘 중에 선택
        # Masking 필요한 label format [3, 10] -> label에도 padding을 붙임
        # labels = tf.placeholder(shape=[None, None], dtype=tf.int32, name="labels_ph")

        # Masking 필요 없는 label format -> [21] -> label에 padding 안 붙이고 shape을 rank 1로 설정 (data_process 쪽에서의 처리가 필요)
        self.labels_ph = tf.placeholder(shape=[None], dtype=tf.int32, name="labels_ph")


    def _inference(self, inputs:tf.Tensor, lengths:tf.Tensor):
        print("Building graph for model: sequence tagger")

        """
        [C]
        단어 임베딩 행렬을 생성합니다.
        단어 id를 단어 임베딩 텐서로 변환합니다.
        """
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

        """
        [D]
        단어 임베딩을 RNN의 입력으로 사용하기 전,
        차원 수를 맞춰주고 성능을 향상시키기 위해
        projection layer를 생성하여 텐서를 통과시킵니다.
        """

        layer_out = tf.layers.dense(
            inputs=layer_out,
            units=self.hparams.rnn_hidden_dim,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.variance_scaling(
                scale=1.0, mode="fan_avg", distribution="normal"),
            name="input_projection"
        )

        """
        [E]
        양방향 RNN을 생성하고, 여기에 텐서를 통과시킵니다.
        이렇게 하여, 단어간 의존 관계가 반영된 단어 자질 텐서를 얻습니다.
        """

        with tf.variable_scope("bi-RNN"):
            # Build RNN layers
            rnn_cell_forward = tf.contrib.rnn.GRUCell(self.hparams.rnn_hidden_dim)
            rnn_cell_backward = tf.contrib.rnn.GRUCell(self.hparams.rnn_hidden_dim)

            # Apply dropout to RNN
            if self.hparams.dropout_keep_prob < 1.0:
                rnn_cell_forward = tf.contrib.rnn.DropoutWrapper(rnn_cell_forward,
                                                                 output_keep_prob=self._dropout_keep_prob_ph)
                rnn_cell_backward = tf.contrib.rnn.DropoutWrapper(rnn_cell_backward,
                                                                  output_keep_prob=self._dropout_keep_prob_ph)

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

        """
        [F]
        마스킹을 적용하여 문장 길이를 통일하기 위해 적용했던 padding을 제거합니다.
        """
        mask = tf.sequence_mask(lengths)
        bi_lstm_out = tf.reshape(tf.boolean_mask(hiddens, mask), [-1, self.hparams.rnn_hidden_dim * 2])
        layer_out = bi_lstm_out  # shape=[sum of seq length, 2*LSTM hidden layer size]


        """
        [G]
        단어 자질 텐서를 바탕으로 단어의 태그를 예측합니다.
        이를 위해 fully-connected(dense) layer를 생성하고 텐서를 통과시킵니다.
        """

        with tf.variable_scope("read-out"):
            prev_layer_size = layer_out.get_shape().as_list()[1]
            weight = tf.get_variable("weight", shape=[prev_layer_size, output_dim],
                                     initializer=tf.initializers.variance_scaling(
                                         scale=2.0, mode="fan_in", distribution="normal"
                                     ))
            bias = tf.get_variable("bias", shape=[output_dim],
                                   initializer=tf.initializers.zeros())
            predictions = tf.add(tf.matmul(layer_out, weight), bias, name='predictions')

        return predictions

    def predict(self, saved_file:str):
        sentence = input("Enter a sentence: ")

        """
        [H]
        입력 문자열을 단어/문장부호 단위로 쪼개고, 이를 다시 단어 id로 변환합니다.
        """
        sentence = word_tokenize(sentence)
        word_ids = []
        for word in sentence:
            if word in self.word2id:
                word_ids.append(self.word2id[word])
            else:
                word_ids.append(len(self.word2id))

        sess = tf.Session()
        with sess.as_default():
            """
            [I]
            태깅을 수행하기 위해 텐서 그래프를 생성합니다.
            """
            dense_word_ids = tf.constant(word_ids)
            lengths = tf.constant(len(word_ids))
            # Insert batch dimension.
            dense_word_ids = tf.expand_dims(dense_word_ids, axis=0)
            lengths = tf.expand_dims(lengths, axis=0)

            with tf.variable_scope("inference", reuse=False):
                logits = self._inference(dense_word_ids, lengths)
            predictions = tf.argmax(logits, axis=1)


            """
            [J]
            저장된 모델을 로드하고, 데이터를 입력하여 태깅 결과를 얻습니다.
            """
            saver = tf.train.Saver()
            saver.restore(sess, saved_file)
            pred_val = sess.run(
                [predictions],
                feed_dict={self._dropout_keep_prob_ph: 1.0}
            )[0]

        """
        [K]
        태깅 결과를 출력합니다.
        """

        pred_str = [self.id2label[i] for i in pred_val]
        for word, tag in zip(sentence, pred_str):
            print("%s[%s]" % (word, tag), end=' ')


    def _load_data(self):
        """
        [L]
        단어->id 및 태그->id 변환 테이블을 텐서 그래프에 추가합니다.
        """

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


        """
        [M]
        입력 데이터 파일을 읽어들여 이를 단어 id로 변환하는 텐서 그래프를 생성합니다.
        """
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

        """
        [N]
        태그 데이터 파일을 읽어들여 이를 태그 id로 변환하는 텐서 그래프를 생성합니다.
        """
        label_dataset = tf.data.TextLineDataset(os.path.join(self.data_dir, "train.labels"))
        batched_label_dataset = label_dataset.batch(self.hparams.batch_size)
        label_iterator = batched_label_dataset.make_initializable_iterator()
        batch_label_str = label_iterator.get_next()
        batch_label = tf.string_split(batch_label_str, " ")
        label_ids = label2id.lookup(batch_label)
        dense_label_ids = tf.sparse_tensor_to_dense(label_ids)
        # shape = [batch_size, time]


        mask = tf.sequence_mask(lengths)
        dense_label_ids = tf.boolean_mask(dense_label_ids, mask)

        self.iterator_initializers.append(input_iterator.initializer)
        self.iterator_initializers.append(label_iterator.initializer)

        return dense_word_ids, dense_label_ids, lengths

    def train(self):
        sess = tf.Session()
        with sess.as_default():
            global_step = tf.Variable(0, name='global_step', trainable=False)

            """tf.Data"""
            # inputs, labels, lengths = self._load_data()
            """placeholders"""
            self._make_placeholders()

            with tf.variable_scope("inference", reuse=False):
                logits = self._inference(self.inputs_ph, self.lengths_ph)

            """
            [O]
            모델을 훈련시키기 위해 필요한 오퍼레이션들을 텐서 그래프에 추가합니다.
            여기에는 loss, train, accuracy 계산 등이 포함됩니다.
            """
            loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels_ph,
                                                                     name="cross_entropy")
            loss_op = tf.reduce_mean(loss_op, name='cross_entropy_mean')
            train_op = tf.train.AdamOptimizer().minimize(loss_op, global_step=global_step)

            eval = tf.nn.in_top_k(logits, self.labels_ph, 1)
            correct_count = tf.reduce_sum(tf.cast(eval, tf.int32))
            accuracy = tf.divide(correct_count, tf.shape(self.labels_ph)[0])


            # Initialize iterators, tables, and variables.
            iterator_initializers = tf.group(*self.iterator_initializers)
            tf.tables_initializer().run()
            tf.global_variables_initializer().run()

            saver = tf.train.Saver()
            total_train_step = math.ceil(len(self.conll_proc.train_examples)/self.hparams.batch_size)
            for epochs_completed in range(self.hparams.num_epochs):
                iterator_initializers.run()
                avg_loss, avg_acc, idx_cnt = 0, 0, 0
                for step in range(total_train_step):

                    """
                    [P]
                    그래프에 데이터를 입력하여 필요한 계산들을 수행하고,
                    Loss에 따라 gradient를 계산하여 파라미터들을 업데이트합니다.
                    이러한 과정을 training step이라고 합니다.
                    """

                    batch_inputs, batch_labels, batch_lengths = \
                        self.conll_proc.get_batch_data(step, self.hparams.batch_size, set_type="train")


                    accuracy_val, loss_val, global_step_val, _ = sess.run(
                        [accuracy, loss_op, global_step, train_op],
                        feed_dict={self.inputs_ph:batch_inputs,
                                   self.labels_ph:batch_labels,
                                   self.lengths_ph:batch_lengths,
                                   self._dropout_keep_prob_ph: self.hparams.dropout_keep_prob}
                    )
                    avg_acc += accuracy_val
                    avg_loss += loss_val
                    idx_cnt += 1
                    if global_step_val % 50 == 0:
                        avg_acc /= idx_cnt
                        avg_loss /= idx_cnt
                        self._logger.info("[Step %d] loss: %.4f, accuracy: %.2f%%" % (
                        global_step_val, avg_loss, avg_acc * 100))
                        avg_loss, avg_acc, idx_cnt = 0, 0 ,0


                """
                [Q]
                전체 학습 데이터에 대하여 1회 학습을 완료하였습니다.
                이를 1 epoch라고 합니다.
                딥러닝 모델의 학습은 일반적으로 수십~수백 epoch 동안 진행됩니다.
                """

                self._logger.info("End of epoch %d." % (epochs_completed + 1))
                save_path = saver.save(sess, "saves/model.ckpt", global_step=global_step_val)
                self._logger.info("Model saved at: %s" % save_path)


def init_logger(path:str):
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    debug_fh = logging.FileHandler(os.path.join(path, "debug.log"))
    debug_fh.setLevel(logging.DEBUG)

    info_fh = logging.FileHandler(os.path.join(path, "info.log"))
    info_fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    info_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
    debug_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s | %(lineno)d:%(funcName)s')

    ch.setFormatter(info_formatter)
    info_fh.setFormatter(info_formatter)
    debug_fh.setFormatter(debug_formatter)

    logger.addHandler(ch)
    logger.addHandler(debug_fh)
    logger.addHandler(info_fh)

    return logger


def train_model(args, builder_class):
    hparams_path = args.hparams

    with open(hparams_path, "r") as f_handle:
        hparams_dict = json.load(f_handle)

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    root_dir = os.path.join(hparams_dict["root_dir"], "%s/" % timestamp)

    logger = init_logger(root_dir)
    logger.info("Loaded hyper-parameter configuration from file: %s" %hparams_path)
    logger.info("Hyper-parameters: %s" %str(hparams_dict))
    hparams_dict["root_dir"] = root_dir

    hparams = collections.namedtuple("HParams", sorted(hparams_dict.keys()))(**hparams_dict)

    with open(os.path.join(root_dir, "hparams.json"), "w") as f_handle:
        json.dump(hparams._asdict(), f_handle, indent=2)

    # Build graph
    model = builder_class(hparams, args.data)
    model.train()


def load_and_predict(args, builder_class):
    hparams_path = args.hparams

    with open(hparams_path, "r") as f_handle:
        hparams_dict = json.load(f_handle)

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    root_dir = os.path.join(hparams_dict["root_dir"], "%s/" % timestamp)

    logger = init_logger(root_dir)
    logger.info("Loaded hyper-parameter configuration from file: %s" %hparams_path)
    logger.info("Hyper-parameters: %s" %str(hparams_dict))
    hparams_dict["root_dir"] = root_dir

    hparams = collections.namedtuple("HParams", sorted(hparams_dict.keys()))(**hparams_dict)

    with open(os.path.join(root_dir, "hparams.json"), "w") as f_handle:
        json.dump(hparams._asdict(), f_handle, indent=2)

    # Build graph
    model = builder_class(hparams, args.data)
    model.predict(args.predict)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Named Entity Tagger.")
    arg_parser.add_argument("--hparams", dest="hparams", required=True,
                            help="Path to the file that contains hyper-parameter settings. (JSON format)")
    arg_parser.add_argument("--data", dest="data", type=str, required=True,
                            help="Directory that contains dataset files.")
    arg_parser.add_argument("--predict", dest="predict", type=str, default=None,
                            help="Path to the saved model.")
    args = arg_parser.parse_args()

    if args.predict is not None:
        load_and_predict(args, SequenceTagger)
    else:
        train_model(args, SequenceTagger)
