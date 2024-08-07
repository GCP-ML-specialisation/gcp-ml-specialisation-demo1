
NUMERICAL_FEATURES = ['trip_miles', 'trip_seconds', "total_arrests", "euclidean"]

# Number of buckets used by tf.transform for encoding each feature.
FEATURE_BUCKET_COUNT = 10

CATEGORICAL_NUMERICAL_FEATURES = [
    'trip_hour', 'trip_day', 'trip_month',
    'pickup_community_area',
    'dropoff_community_area'
]

CATEGORICAL_STRING_FEATURES = [
    'payment_type', "loc_cross"
]

# Number of vocabulary terms used for encoding categorical features.
VOCAB_SIZE = 1000

# Count of out-of-vocab buckets in which unrecognized categorical are hashed.
OOV_SIZE = 10

# Keys
LABEL_KEY = 'tips'
FARE_KEY = 'fare'

def t_name(key):
  """
  Rename the feature keys so that they don't clash with the raw keys when
  running the Evaluator component.
  Args:
    key: The original feature key
  Returns:
    key with '_xf' appended
  """
  return key + '_xf'
