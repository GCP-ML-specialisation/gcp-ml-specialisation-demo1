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
   - Execute the `run_pipeline.py` file to initiate the pipeline. A console log link will appear in the terminal, allowing you to monitor the pipeline's execution.
