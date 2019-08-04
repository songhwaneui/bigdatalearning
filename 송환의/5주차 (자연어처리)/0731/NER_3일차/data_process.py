import os
import numpy as np


class InputExample(object):
	"""A single training/test example for simple sequence classification."""

	def __init__(self, input_text, label):
		self.input_text = input_text
		self.label = label


class CoNLL2003Processor(object):
  """
  This code is implemented based on:
  https://github.com/google-research/bert/blob/master/run_classifier.py
  """
  def __init__(self, hparams=None, word2id=None, label2id=None):
    self.hparams = hparams
    self.word2id = word2id
    self.label2id = label2id
    self.train_examples = self._create_examples(self._get_train_examples())

  def _get_train_examples(self):
    with open(os.path.join("./CoNLL-2003/train.inputs"), "r") as f_handle:
      train_data = [line.rstrip() for line in f_handle if len(line.rstrip()) > 0]
      input_word_tok = []
      for sentence in train_data:
          input_word_tok.append(sentence.split(" "))
      print("train_inputs", len(input_word_tok))
      print("train_inputs sample", input_word_tok[0:10])

    with open(os.path.join("./CoNLL-2003/train.labels"), "r") as f_handle:
      train_labels = [line.rstrip() for line in f_handle if len(line.rstrip()) > 0]
      label_word_tok = []
      for label_sentence in train_labels:
          label_word_tok.append(label_sentence.split(" "))
      print("train_labels", len(label_word_tok))
      print("train_labels sample", label_word_tok[0:10])

    train_data = np.stack([input_word_tok, label_word_tok], axis=1)
    print("[inputs, labels] stacked data : ", len(train_data))

    return train_data

  def _create_examples(self, inputs, set_type="train"):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, input_data) in enumerate(inputs):
      input = input_data[0]
      label = input_data[1]
      # print(input, label)
      examples.append(
        InputExample(input_text=input, label=label))

    print("%s data creation is finished! %d" % (set_type, len(examples)))
    print("total number of examples ", len(examples))
    return examples

  def get_batch_data(self, curr_index, batch_size, set_type="train"):
    inputs = []
    lengths = []
    label_ids = []

    example = self.train_examples

    for index, each_example in enumerate(example[curr_index * batch_size:batch_size * (curr_index + 1)]):
      input_id, label_id, input_length = \
        self.convert_single_example(each_example, word2id=self.word2id, label2id=self.label2id)
      # print(input_id, label_id, input_length)

      inputs.append(input_id)
      label_ids.extend(label_id)
      lengths.append(input_length)

    pad_inputs = self._rank_2_pad_process(inputs)

    return pad_inputs, label_ids, lengths

  def _rank_2_pad_process(self, inputs, special_id=False):
    append_id = 0
    if special_id:
      append_id = -1  # user_id -> -1

    max_sent_len = 0
    for sent in inputs:
      max_sent_len = max(len(sent), max_sent_len)

    # print("text_a max_lengths in a batch", max_sent_len)
    padded_result = []
    sent_buffer = []
    for sent in inputs:
      for i in range(max_sent_len - len(sent)):
        sent_buffer.append(append_id)
      sent.extend(sent_buffer)
      padded_result.append(sent)
      sent_buffer = []

    # padded_result = np.array(padded_result)

    return padded_result

  def convert_single_example(self, example, word2id, label2id):
    # input
    tokenized_input = example.input_text
    # print(tokenized_input)

    tokenized_input_id = [0] * len(example.input_text)
    for a_idx, inp_token_a in enumerate(tokenized_input):
      try:
        tokenized_input_id[a_idx] = word2id[inp_token_a]
      except KeyError:
        tokenized_input_id[a_idx] = word2id["<UNK>"]

    # label
    tokenized_label = example.label
    tokenized_label_id = [0] * len(example.label)
    for a_idx, lab_token_a in enumerate(tokenized_label):
      tokenized_label_id[a_idx] = label2id[lab_token_a]

    input_length = len(tokenized_input_id)

    assert len(tokenized_input_id) == len(tokenized_input_id)

    return tokenized_input_id, tokenized_label_id, input_length

if __name__ == '__main__':
    conll_processor = CoNLL2003Processor()
