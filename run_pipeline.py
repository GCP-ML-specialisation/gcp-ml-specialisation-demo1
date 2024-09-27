from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs
import logging
logging.getLogger().setLevel(logging.INFO)

from src.utils import * 
import os
import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

GOOGLE_CLOUD_PROJECT = config["project"]
GOOGLE_CLOUD_REGION = config["region"]
GCS_BUCKET_NAME = config["gcs_location"]  

PIPELINE_NAME = config['pipeline_name']
EXPERIMENT_NAME = config['experiment_name']

# Path to various pipeline artifact.
PIPELINE_ROOT = 'gs://{}/{}/pipeline_root/{}'.format(GCS_BUCKET_NAME, EXPERIMENT_NAME, PIPELINE_NAME)

# Paths for users' Python module.
MODULE_ROOT = 'gs://{}/{}/pipeline_module/{}'.format(GCS_BUCKET_NAME, EXPERIMENT_NAME, PIPELINE_NAME)

# Paths for users' data.
DATA_ROOT = 'gs://{}/{}/data/{}'.format(GCS_BUCKET_NAME, EXPERIMENT_NAME, PIPELINE_NAME)

# Name of Vertex AI Endpoint.
ENDPOINT_NAME = 'prediction-' + PIPELINE_NAME

print('PIPELINE_ROOT: {}'.format(PIPELINE_ROOT))

_taxi_transform_module_file = 'taxi_trainer.py'
_taxi_trainer_module_file = 'taxi_transform.py'

PIPELINE_DEFINITION_FILE = PIPELINE_NAME + '_pipeline.json'

runner = tfx.orchestration.experimental.KubeflowV2DagRunner(
    config=tfx.orchestration.experimental.KubeflowV2DagRunnerConfig(),
    output_filename=PIPELINE_DEFINITION_FILE)
_ = runner.run(
    _create_pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        data_root=DATA_ROOT,
        transform_file=os.path.join(MODULE_ROOT, _taxi_transform_module_file),
        trainer_file=os.path.join(MODULE_ROOT, _taxi_trainer_module_file),
        endpoint_name=ENDPOINT_NAME,
        project_id=GOOGLE_CLOUD_PROJECT,
        region=GOOGLE_CLOUD_REGION,
        # We will use CPUs only for now.
        use_gpu=False,
        eval_config= eval_config))


aiplatform.init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_REGION)

job = pipeline_jobs.PipelineJob(template_path=PIPELINE_DEFINITION_FILE,
                                enable_caching=True,
                                display_name=PIPELINE_NAME)
job.submit()

