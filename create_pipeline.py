import os
import apache_beam as beam
import tensorflow as tf
import tensorflow_transform as tft
import tfx
from tfx.components import ImportExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Trainer, Evaluator, Pusher
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing
from tfx.dsl.components.common import resolver
from tfx.proto import pusher_pb2

from tfx.orchestration import pipeline
from tfx.orchestration.local.local_dag_runner import LocalDagRunner  # Use KubeflowDagRunner or VertexDagRunner for Kubeflow or Vertex AI
from google.cloud import aiplatform
import pprint as pp

import tensorflow_model_analysis as tfma

# Import correct TrainArgs and EvalArgs classes
from tfx.proto import trainer_pb2 as trainer_pb
from tfx.proto import evaluator_pb2 as evaluator_pb

from ml_metadata.proto import metadata_store_pb2
from google.cloud import aiplatform


import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

raw_data_query = config["query"]
_taxi_transform_module_file = 'taxi_transform.py'
_taxi_trainer_module_file = 'taxi_trainer.py'
project = config["project"]
gcs_location = config["gcs_location"]  
pipeline_name = config["pipeline_name"]
pipeline_root = config["pipeline_root"]
data_path = config["data_path"]
transform_module_file = os.path.abspath(_taxi_transform_module_file)
trainer_module_file = os.path.abspath(_taxi_trainer_module_file)
serving_model_dir = config["serving_model_dir"]

