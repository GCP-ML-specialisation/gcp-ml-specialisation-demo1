# GCP ML Specialisation Demo 1

To set up and run the demo, follow these steps:

1. **Clone the repository**.

2. **Update the configuration file**:
   - Modify the `config.yaml` as required.

3. **Set up Google Cloud configuration**:
   - Set your project ID:
     ```bash
     PROJECT_ID="<your_project_id>"
     gcloud config set project ${PROJECT_ID}
     ```
   - Set your account ID:
     ```bash
     ACCOUNT_ID="<your_account_id>"
     gcloud config set account ${ACCOUNT_ID}
     ```
   - Create a new configuration:
     ```bash
     CONFIG="<your_config_name>"
     gcloud config configurations create ${CONFIG}
     ```
   - Authenticate with Google Cloud:
     ```bash
     gcloud auth application-default login
     gcloud auth login
     ```

4. **Set up the service account**:
   - Use the following service account:
     ```bash
     SERVICE_ACCOUNT="<your service account>"
     ```

5. **Run the pipeline**:
   - Create Root Directory for the Experiment:
   Use the following command to create a root directory in your GCP bucket:
   
   ```bash
   !gsutil mkdir "gs://{project}/{experiment_name}/{pipeline_name}/data" 
  ```
   - Copy Local File to GCP Bucket:
   Use the following command to copy it to your GCP bucket:
   
   ```bash
   DATA_PATH = "data/chicago_trips.csv"   
   !gsutil cp {DATA_PATH} gs://{project}/{experiment_name}/{pipeline_name}/data/data.csv
  ```
   - Copy Module Files to GCP Bucket:
   ```bash 
   MODULE_ROOT = gs://{bucket}/{experiment_name}/{pipeline_name}
   taxi_transform_module_file="src/taxi_transform.py"
   taxi_trainer_module_file="src/taxi_trainer.py"
   gsutil cp ${taxi_transform_module_file} ${MODULE_ROOT}/
   gsutil cp ${taxi_trainer_module_file} ${MODULE_ROOT}/   
   ```
   - Execute the `run_pipeline.py` file to initiate the pipeline. 
  