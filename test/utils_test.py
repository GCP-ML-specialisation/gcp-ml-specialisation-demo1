from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow.python.lib.io import file_io
from tensorflow_transform import coders as tft_coders
from tensorflow_transform.tf_metadata import schema_utils


# Tf.Transform considers these features as "raw"
def _get_raw_feature_spec(schema):
  return schema_utils.schema_as_feature_spec(schema).feature_spec

def _make_proto_coder(schema):
  raw_feature_spec = _get_raw_feature_spec(schema)
  raw_schema = schema_utils.schema_from_feature_spec(raw_feature_spec)
  return tft_coders.ExampleProtoCoder(raw_schema)

def _read_schema(path):
  """Reads a schema from the provided location.
  Args:
    path: The location of the file holding a serialized Schema proto.
  Returns:
    An instance of Schema or None if the input argument is None
  """
  result = schema_pb2.Schema()
  contents = file_io.read_file_to_string(path)
  text_format.Parse(contents, result)
  return result

def _make_proto_coder(schema):
  raw_feature_spec = _get_raw_feature_spec(schema)
  raw_schema = schema_utils.schema_from_feature_spec(raw_feature_spec)
  return tft_coders.ExampleProtoCoder(raw_schema)