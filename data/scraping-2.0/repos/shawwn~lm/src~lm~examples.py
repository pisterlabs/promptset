"The only valid example formats accepted by the framework"

import collections

import numpy as np
import tensorflow as tf

import farmhash

CONTENT_KEY = "content"
TARGET_KEY = "target"

PreProcessedTextLine = collections.namedtuple(
    "PreProcessedTextLine", ["id", "content", "target", "offset_start", "offset_end"]
)

Seq2SeqSimpleExample = collections.namedtuple(
    "Seq2SeqSimpleExample", [CONTENT_KEY, TARGET_KEY]
)


def _uint64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=np.int64(np.array(value, dtype=np.uint64)))
    )


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_example(example_proto) -> dict:
    features = {
        "id": tf.io.VarLenFeature(tf.int64),
        "content": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.VarLenFeature(tf.int64),
        "offset_start": tf.io.VarLenFeature(tf.int64),
        "offset_end": tf.io.VarLenFeature(tf.int64),
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    return {
        "id": tf.cast(parsed_features["id"], tf.uint64),
        "content": parsed_features["content"],
        # WARNING: remapping from target to targets
        "targets": tf.sparse.to_dense(tf.cast(parsed_features["target"], tf.int64)),
        "offset_start": tf.sparse.to_dense(
            tf.cast(parsed_features["offset_start"], tf.uint64)
        ),
        "offset_end": tf.sparse.to_dense(
            tf.cast(parsed_features["offset_end"], tf.uint64)
        ),
    }


# def read_example(example_proto, max_seq_len=1024) -> dict:
#     features = {
#         "id": tf.VarLenFeature(tf.uint64, default=-1),
#         "content": tf.VarLenFeature(tf.bytes, default=0),
#         "target": tf.VarLenFeature(tf.uint64, default=0),
#         "offset_start": tf.VarLenFeature(tf.uint64, default=0),
#         "offset_end": tf.VarLenFeature(tf.uint64, default=0),
#     }
#     return tf.parse_single_example(example_proto, features)


def create_example(features: PreProcessedTextLine) -> tf.train.Example:
    feature = {
        "id": _uint64_feature([features.id]),
        #"content": _bytes_feature(features.content.encode("utf-8")),
        "text": _uint64_feature(features.target),
        "offset_start": _uint64_feature(features.offset_start),
        "offset_end": _uint64_feature(features.offset_end),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def transform_many_and_write_one_tfrecord(job):
    tokenizer, sources, dst, args = job
    token_count = 0
    example_count = 0
    with tf.io.TFRecordWriter(dst) as w:
        for source in sources:
            print(str(source))
            for uids, sources, tokens, start_offsets, end_offsets in batch_tokenizer(tokenizer, source, by_line=args.by_line):
                if len(tokens) > 0:
                    result = PreProcessedTextLine(uids, sources, tokens, start_offsets, end_offsets)
                    example = create_example(result)
                    w.write(example.SerializeToString())
                    token_count += len(tokens)
                    example_count += 1
    return token_count, example_count


import numpy as np
import io

def tokens_to_buffer(chunks, stride):
  assert stride in [2, 4]
  tokens = np.array(chunks, dtype=np.uint16 if stride == 2 else np.int32)
  return tokens.tobytes()

def tokens_from_buffer(data, stride):
  assert stride in [2, 4]
  return np.frombuffer(data, dtype=np.uint16 if stride == 2 else np.int32)

def tokens_to_file(out, chunks, stride):
  if isinstance(out, tf.io.gfile.GFile):
    data = tokens_to_buffer(chunks, stride)
    out.write(data)
  else:
    assert stride in [2, 4]
    tokens = np.array(chunks, dtype=np.uint16 if stride == 2 else np.int32)
    tokens.tofile(out)

def tokens_from_file(f, stride):
  if isinstance(f, tf.io.gfile.GFile):
    return tokens_from_buffer(f.read(), stride)
  if isinstance(f, str) and f.startswith('gs://'):
    with tf.io.gfile.GFile(f, 'rb') as f:
      return tokens_from_file(f, stride)
  assert stride in [2, 4]
  return np.fromfile(f, dtype=np.uint16 if stride == 2 else np.int32)


import tqdm

def transform_many_and_write_one_tok16_or_tok32(job):
    tokenizer, sources, dst, args = job
    assert args.format in ["tok16", "tok32"]
    bytes_per_token = 2 if args.format == "tok16" else 4
    token_count = 0
    example_count = 0
    ftfy = False if args.no_ftfy else True
    eos = 50256 if args.encoder == 'gpt2' else 0
    # # Should we warn about this?
    # if eos != 0:
    #   print('Using EOS 0')
    with open(dst, 'wb') as w:
        pbar = tqdm.tqdm(sources)
        for source in pbar:
            pbar.set_description(source)
            for uids, sources, tokens, start_offsets, end_offsets in batch_tokenizer(tokenizer, source, by_line=args.by_line, ftfy=ftfy):
                if len(tokens) > 0:
                    tokens.append(eos)
                    tokens_to_file(w, tokens, stride=bytes_per_token)
                    token_count += len(tokens)
                    example_count += 1
    return token_count, example_count


def ftfy_text(text):
  from ftfy import fix_text
  fixed = fix_text(text)
  # replace unicode … with ... which ftfy doesn't do by default
  # NOTE: this departs from openai's convention of calling
  # ftfy.fix_text() with default arguments. In particular,
  # OpenAI's GPT-2 models do generate unicode ellipses.
  # Nonetheless, we replace unicdoe ellipses with ... to
  # increase the chances of semantic understanding.
  fixed = fixed.replace(' …', '...') # first pass: convert "foo  …" to "foo..."
  #fixed = fixed.replace(' …', '...') # second pass: convert "foo …" to "foo..."
  fixed = fixed.replace('…', '...') # final pass: convert "foo…" to "foo..."
  return fixed


def batch_tokenizer(tokenizer, txtfile_location, by_line=False, ftfy=True):
    # just convert to the token ids, we will do adaptative padding on training time.
    with tf.io.gfile.GFile(txtfile_location, "rb") as f:
      if by_line:
        sources = [l.decode("utf-8") for l in f.readlines()]
      else:
        sources = [f.read().decode("utf-8")]
    if len(sources) <= 0:
      # tokenizer crashes when given an empty list, so give it an empty string
      # (this happens in --by_line mode for empty files)
      sources = ['']
    if ftfy:
      sources = [ftfy_text(source) for source in sources]
    uids = [farmhash.fingerprint64(source) for source in sources]
    batches = tokenizer.batch_encode_plus(
        sources,
        return_token_type_ids=True,
        pad_to_max_length=False,
        truncation=False,
        add_special_tokens=True,
        return_offsets_mapping=True,
        verbose=False,
    )

    return zip(
        uids,
        sources,
        batches["input_ids"],
        [[start for start, end in offsets] for offsets in batches["offset_mapping"]],
        [[end for start, end in offsets] for offsets in batches["offset_mapping"]],
    )


def _int64_list_feature(int_list):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=int_list))


def gen_serialization(ndigit):
    def serialize(tokens, idx):
        """
        Creates a tf.Example message ready to be written to a file.
        """

        # Create a dictionary mapping the feature name to the tf.Example-compatible
        # data type.
        feature = {
            "tokens": _int64_list_feature(tokens),
            "idx": _int64_list_feature(idx),
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        return example.SerializeToString()

    feature_spec = {
        "tokens": tf.io.FixedLenFeature([ndigit * 3], dtype=tf.dtypes.int64),
        "idx": tf.io.FixedLenFeature([ndigit * 3], dtype=tf.dtypes.int64),
    }

    def deserialize(example):
        return tf.io.parse_single_example(example, features=feature_spec)

    return serialize, deserialize


def _serialize_seq2seq(self):
    feature = {
        CONTENT_KEY: _int64_list_feature(self.content),
        TARGET_KEY: _int64_list_feature(self.target),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    return example.SerializeToString()
    # raise ValueError('type %r not yet supported' % type(ex))
    # feature_spec = {
    #     "tokens": tf.io.FixedLenFeature([ndigit * 3], dtype=tf.dtypes.int64),
    #     "idx": tf.io.FixedLenFeature([ndigit * 3], dtype=tf.dtypes.int64),
    # }


Seq2SeqSimpleExample.serialize = _serialize_seq2seq
