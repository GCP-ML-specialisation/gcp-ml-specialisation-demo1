import tensorflow as tf
from tfx import v1 as tfx
import kfp
import tensorflow_model_analysis as tfma

from tfx.components import Evaluator, Pusher
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing
from tfx.dsl.components.common import resolver


eval_config = tfma.EvalConfig(
    model_specs=[
        tfma.ModelSpec(
            signature_name='serving_default',
            label_key='tips',
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
                      change_threshold=tfma.GenericChangeThreshold(
                          direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                          absolute={'value': -1e-10})))
            ]
        )
    ],
    slicing_specs=[
        tfma.SlicingSpec(),
        tfma.SlicingSpec(
            feature_keys=['trip_hour'])
    ])

def _create_pipeline(pipeline_name: str, pipeline_root: str, data_root: str,
                     transform_file: str,
                     trainer_file: str, endpoint_name: str, project_id: str,
                     region: str,
                     use_gpu: bool,
                     eval_config) -> tfx.dsl.Pipeline:
  """Implements the penguin pipeline with TFX."""
  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = tfx.components.CsvExampleGen(input_base=data_root)

  vertex_job_spec = {
      'project': project_id,
      'worker_pool_specs': [{
          'machine_spec': {
              'machine_type': 'n1-standard-4',
          },
          'replica_count': 1,
          'container_spec': {
              'image_uri': 'gcr.io/tfx-oss-public/tfx:{}'.format(tfx.__version__),
          },
      }],
  }
  if use_gpu:
    vertex_job_spec['worker_pool_specs'][0]['machine_spec'].update({
        'accelerator_type': 'NVIDIA_TESLA_K80',
        'accelerator_count': 1
    })

  statistics_gen = tfx.components.StatisticsGen(
    examples=example_gen.outputs['examples'])
  vertex_job_spec = {
      'project': project_id,
      'worker_pool_specs': [{
          'machine_spec': {
              'machine_type': 'n1-standard-4',
          },
          'replica_count': 1,
          'container_spec': {
              'image_uri': 'gcr.io/tfx-oss-public/tfx:{}'.format(tfx.__version__),
          },
      }],
  }
  if use_gpu:
    vertex_job_spec['worker_pool_specs'][0]['machine_spec'].update({
        'accelerator_type': 'NVIDIA_TESLA_K80',
        'accelerator_count': 1
    })

  schema_gen = tfx.components.SchemaGen(
    statistics=statistics_gen.outputs['statistics'],
    infer_feature_shape=False)
  vertex_job_spec = {
      'project': project_id,
      'worker_pool_specs': [{
          'machine_spec': {
              'machine_type': 'n1-standard-4',
          },
          'replica_count': 1,
          'container_spec': {
              'image_uri': 'gcr.io/tfx-oss-public/tfx:{}'.format(tfx.__version__),
          },
      }],
  }

  example_validator = tfx.components.ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=schema_gen.outputs['schema'])
  vertex_job_spec = {
      'project': project_id,
      'worker_pool_specs': [{
          'machine_spec': {
              'machine_type': 'n1-standard-4',
          },
          'replica_count': 1,
          'container_spec': {
              'image_uri': 'gcr.io/tfx-oss-public/tfx:{}'.format(tfx.__version__),
          },
      }],
  }

  if use_gpu:
    vertex_job_spec['worker_pool_specs'][0]['machine_spec'].update({
        'accelerator_type': 'NVIDIA_TESLA_K80',
        'accelerator_count': 1
    })

  transform = tfx.components.Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file=transform_file)
  vertex_job_spec = {
      'project': project_id,
      'worker_pool_specs': [{
          'machine_spec': {
              'machine_type': 'n1-standard-4',
          },
          'replica_count': 1,
          'container_spec': {
              'image_uri': 'gcr.io/tfx-oss-public/tfx:{}'.format(tfx.__version__),
          },
      }],
  }

  if use_gpu:
    vertex_job_spec['worker_pool_specs'][0]['machine_spec'].update({
        'accelerator_type': 'NVIDIA_TESLA_K80',
        'accelerator_count': 1
    })

  # Trains a model using Vertex AI Training.
  trainer = tfx.extensions.google_cloud_ai_platform.Trainer(
      module_file=trainer_file,
      # examples=example_gen.outputs['examples'],
      examples=transform.outputs['transformed_examples'],
      transform_graph=transform.outputs['transform_graph'],
      schema=schema_gen.outputs['schema'],
      train_args=tfx.proto.TrainArgs(num_steps=100),
      eval_args=tfx.proto.EvalArgs(num_steps=5),
      custom_config={
          tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY:
              True,
          tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY:
              region,
          tfx.extensions.google_cloud_ai_platform.TRAINING_ARGS_KEY:
              vertex_job_spec,
          'use_gpu':
              use_gpu,
      })

  vertex_serving_spec = {
      'project_id': project_id,
      'endpoint_name': endpoint_name,
      'machine_type': 'n1-standard-4',
  }

  serving_image = 'us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-6:latest'
  if use_gpu:
    vertex_serving_spec.update({
        'accelerator_type': 'NVIDIA_TESLA_K80',
        'accelerator_count': 1
    })
    serving_image = 'us-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-6:latest'

  model_resolver = resolver.Resolver(
      strategy_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
      model=Channel(type=Model),
      model_blessing=Channel(
          type= ModelBlessing)).with_id('latest_blessed_model_resolver')

  vertex_serving_spec = {
      'project_id': project_id,
      'endpoint_name': endpoint_name,
      'machine_type': 'n1-standard-4',
  }

  serving_image = 'us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-6:latest'
  if use_gpu:
    vertex_serving_spec.update({
        'accelerator_type': 'NVIDIA_TESLA_K80',
        'accelerator_count': 1
    })
    serving_image = 'us-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-6:latest'

  # Evaluator component
  evaluator = Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      baseline_model=model_resolver.outputs['model'],
      eval_config=eval_config)

  vertex_serving_spec = {
      'project_id': project_id,
      'endpoint_name': endpoint_name,
      'machine_type': 'n1-standard-4',
  }

  serving_image = 'us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-6:latest'
  if use_gpu:
    vertex_serving_spec.update({
        'accelerator_type': 'NVIDIA_TESLA_K80',
        'accelerator_count': 1
    })
    serving_image = 'us-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-6:latest'


  pusher = tfx.extensions.google_cloud_ai_platform.Pusher(
      model=trainer.outputs['model'],
      custom_config={
          tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY:
              True,
          tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY:
              region,
          tfx.extensions.google_cloud_ai_platform.VERTEX_CONTAINER_IMAGE_URI_KEY:
              serving_image,
          tfx.extensions.google_cloud_ai_platform.SERVING_ARGS_KEY:
            vertex_serving_spec,
      })

  components = [
      example_gen,
      statistics_gen,
      schema_gen,
      transform,
      trainer,
      model_resolver,
      evaluator,
      pusher,
  ]

  return tfx.dsl.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=components)