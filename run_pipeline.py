from src.utils import * 
from datetime import datetime
from tfx.orchestration.local.local_dag_runner import LocalDagRunner

import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

project = config["project"]
gcs_location = config["gcs_location"]  
serving_model_dir = config["serving_model_dir"]
region = config["region"]
pipeline_name = config["pipeline_name"]
pipeline_root = config["pipeline_root"]
data_path = config["data_path"]
base_model_name = config["base_model_name"]

_taxi_transform_module_file = 'src/taxi_transform.py'
_taxi_trainer_module_file = 'src/taxi_trainer.py'
transform_module_file = os.path.abspath(_taxi_transform_module_file)
trainer_module_file = os.path.abspath(_taxi_trainer_module_file)

current_date = datetime.now().strftime("%d%m%y")
serving_model_dir = f"{serving_model_dir}{current_date}/"

MLMD_SQLLITE = "mlmd.sqllite"
metadata_connection_config = metadata_store_pb2.ConnectionConfig()
metadata_connection_config.sqlite.filename_uri = MLMD_SQLLITE
metadata_connection_config.sqlite.connection_mode = 3

def main():
    run_beam_pipeline()
    tfx_pipeline = create_pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        data_path=data_path,
        transform_module_file=transform_module_file,
        trainer_module_file=trainer_module_file,
        serving_model_dir=serving_model_dir,
        project_id=project,
        metadata_connection_config = metadata_connection_config
    )
    LocalDagRunner().run(tfx_pipeline)
    deploy_model(base_model_name, project, region, gcs_location, serving_model_dir)

if __name__ == "__main__":
    main()    