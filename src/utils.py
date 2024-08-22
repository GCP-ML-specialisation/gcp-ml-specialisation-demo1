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
import taxi_constants
from datetime import datetime


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

current_date = datetime.now().strftime("%d%m%Y")
serving_model_dir = f"{serving_model_dir}{current_date}"


args ={
    "project": project,
    "gcs_location": gcs_location,
    "raw_data_query": raw_data_query
}

pipeline_options = beam.pipeline.PipelineOptions(flags=[], **args)

def parse_bq_record(bq_record):
    features = {}
    for key, value in bq_record.items():
        if isinstance(value, int):
            features[key] = tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
        elif isinstance(value, float):
            features[key] = tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
        elif isinstance(value, str):
            features[key] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))
        elif value is None:
            features[key] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'NA']))
        else:
            raise ValueError(f"Unsupported data type for key: {key}, value: {value}")

    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example

def run_beam_pipeline():
    with beam.Pipeline(options=pipeline_options) as pipeline:
        (pipeline
         | "ReadFromBigQuery" >> beam.io.ReadFromBigQuery(
                        query=args["raw_data_query"],
                        project=args["project"],
                        use_standard_sql=True,
                        gcs_location=args["gcs_location"],
         )
         | "ParseData" >> beam.Map(parse_bq_record)
         | "WriteToTFRecord" >> beam.io.WriteToTFRecord(
             f'gs://{args["project"]}/tfrecords/data',
             file_name_suffix='.tfrecord',
             coder=beam.coders.ProtoCoder(tf.train.Example)
         )
        )

eval_config = tfma.EvalConfig(
    model_specs=[
        tfma.ModelSpec(
            signature_name='serving_default',
            label_key=taxi_constants.LABEL_KEY,
            preprocessing_function_names=['transform_features'],
            )
        ],
    metrics_specs=[
        tfma.MetricsSpec(
            metrics=[
                tfma.MetricConfig(class_name='ExampleCount'),
                tfma.MetricConfig(class_name='BinaryAccuracy',
                  threshold=tfma.MetricThreshold(
                      value_threshold=tfma.GenericValueThreshold(
                          lower_bound={'value': 0.5}),
                      # Change threshold will be ignored if there is no
                      # baseline model resolved from MLMD (first run).
                      change_threshold=tfma.GenericChangeThreshold(
                          direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                          absolute={'value': -1e-10})))
            ]
        )
    ],
    slicing_specs=[
        # An empty slice spec means the overall slice, i.e. the whole dataset.
        tfma.SlicingSpec(),
        # Data can be sliced along a feature column. In this case, data is
        # sliced along feature column trip_start_hour.
        tfma.SlicingSpec(
            feature_keys=['trip_hour'])
    ])        

def create_pipeline(pipeline_name, pipeline_root, data_path, transform_module_file, trainer_module_file, serving_model_dir, project_id, metadata_connection_config) :

    # ExampleGen component
    example_gen = ImportExampleGen(input_base=data_path)

    # StatisticsGen component
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

    # SchemaGen component
    schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'], infer_feature_shape=False)

    # ExampleValidator component
    example_validator = ExampleValidator(statistics=statistics_gen.outputs['statistics'], schema=schema_gen.outputs['schema'])

    # Transform component
    transform = Transform(examples=example_gen.outputs['examples'],
                          schema=schema_gen.outputs['schema'],
                          module_file=transform_module_file)

    # Trainer component
    trainer = Trainer(module_file=trainer_module_file,
                      examples=transform.outputs['transformed_examples'],
                      transform_graph=transform.outputs['transform_graph'],
                      schema=schema_gen.outputs['schema'],
                      train_args=trainer_pb.TrainArgs(num_steps=10000),
                      eval_args=trainer_pb.EvalArgs(num_steps=5000))

    # Resolver to find the latest blessed model
    model_resolver = resolver.Resolver(
        strategy_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
        model=Channel(type=Model),
        model_blessing=Channel(
            type=ModelBlessing)).with_id('latest_blessed_model_resolver')

    # Evaluator component
    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config)

    # Pusher component
    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(base_directory=serving_model_dir)))

    # Assemble the pipeline
    components = [
        example_gen, statistics_gen, schema_gen, example_validator,
        transform, trainer, model_resolver, evaluator, pusher
    ]

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata_connection_config
    )

def deploy_model(model_display_name, project, location, bucket, serving_model_dir):
    
    # Initialize Vertex AI client
    aiplatform.init(project=project, location=location)
        
    artifact_uri = get_artifact_uri(bucket, serving_model_dir)
    
    model_display_name = get_next_model_name(model_display_name, project, location)

    # Upload the model
    model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-5:latest'
    )

    # Deploy the model to an endpoint
    endpoint = model.deploy(machine_type='n1-standard-4')

    endpoint_uri = endpoint.resource_name
    print("Endpoint URI:", endpoint_uri)

def get_artifact_uri(bucket, serving_model_dir):
    from google.cloud import storage
    import re

    # Initialize Google Cloud Storage client
    client = storage.Client()

    # Define your bucket and model directory
    bucket_name = bucket
    model_directory = "/".join(serving_model_dir.split("/")[3:])

    # Get the bucket
    bucket = client.get_bucket(bucket_name)

    # List all blobs in the model directory
    blobs = bucket.list_blobs(prefix=model_directory)

    # Extract model versions
    model_versions = []
    for blob in blobs:
        # Find directories within the model directory
        match = re.match(rf"{model_directory}(\d+)/$", blob.name)
        if match:
                model_versions.append(match.group(1))
    if not model_versions:
        raise ValueError("No model versions found in the specified directory.")
    
    # Sort model versions to find the latest one
    latest_model_version = sorted(model_versions)[-1]

    latest_model_dir = f"{serving_model_dir}{latest_model_version}/"

    return latest_model_dir

def get_next_model_name(base_model_name, project, location):
    # Initialize the Vertex AI client
    aiplatform.init(project=project, location=location)

    # List models in the specified project and location
    models = aiplatform.Model.list()

    # Check if there are any models
    if not models:
        # If no models exist, start with version 01
        new_model_name = f"{base_model_name}-v01"
        print(f"No existing models found. Starting with model name: {new_model_name}")
        return new_model_name

    # Sort models by creation time in descending order
    latest_model = max(models, key=lambda model: model.create_time)

    # Extract the display name of the latest model
    latest_model_name = latest_model.display_name

    # Use regex to extract the base name and version number
    match = re.match(rf"({re.escape(base_model_name)}-v)(\d+)", latest_model_name)
    
    if match:
        base_name = match.group(1)
        version_number = int(match.group(2))
        # Increment the version number by 1
        new_version_number = version_number + 1
        new_model_name = f"{base_name}{new_version_number:02d}"
    else:
        # If the regex doesn't match, start from version 01
        new_model_name = f"{base_model_name}-v01"

    print(f"Latest model name: {latest_model_name}")
    print(f"New model name: {new_model_name}")

    return new_model_name