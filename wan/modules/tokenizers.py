import html

import ftfy
import regex as re
from transformers import AutoTokenizer

__all__ = ['HuggingfaceTokenizer']


def basic_clean(text):
  text = ftfy.fix_text(text)
  text = html.unescape(html.unescape(text))
  return text.strip()


def whitespace_clean(text):
  text = re.sub(r'\s+', ' ', text)
  text = text.strip()
  return text


class HuggingfaceTokenizer:
  def __init__(self, name, seq_len=None, **kwargs):
    self.name = name
    self.seq_len = seq_len

    # init tokenizer
    self.tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)

  def __call__(self, sequence, **kwargs):
    return_mask = kwargs.pop('return_mask', False)

    # arguments
    _kwargs = {'return_tensors': 'pt'}
    if self.seq_len is not None:
      _kwargs.update({'padding': 'max_length', 'truncation': True, 'max_length': self.seq_len})
    _kwargs.update(**kwargs)

    # tokenization
    if isinstance(sequence, str):
      sequence = [sequence]

    sequence = [whitespace_clean(basic_clean(u)) for u in sequence]
    ids = self.tokenizer(sequence, **_kwargs)

    # output
    if return_mask:
      return ids.input_ids, ids.attention_mask
    else:
      return ids.input_ids
