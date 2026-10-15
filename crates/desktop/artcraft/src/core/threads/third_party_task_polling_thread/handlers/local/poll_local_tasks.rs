use crate::core::providers::local_image_server::local_image_server_client::{
  poll_local_image_job, LocalImageServerError, LocalJobState,
};
use crate::core::state::app_env_configs::app_env_configs::AppEnvConfigs;
use crate::core::state::data_dir::app_data_root::AppDataRoot;
use crate::core::state::task_database::TaskDatabase;
use crate::core::threads::third_party_task_polling_thread::handlers::local::handle_local_complete::handle_local_complete;
use crate::core::threads::third_party_task_polling_thread::handlers::local::handle_local_failure::handle_local_failure;
use crate::services::storyteller::state::storyteller_credential_manager::StorytellerCredentialManager;
use log::{info, warn};
use sqlite_tasks::queries::task::Task;
use tauri::AppHandle;

pub async fn poll_local_tasks(
  app_handle: &AppHandle,
  app_env_configs: &AppEnvConfigs,
  app_data_root: &AppDataRoot,
  task_database: &TaskDatabase,
  storyteller_creds_manager: &StorytellerCredentialManager,
  local_tasks: &[&Task],
) {
  for task in local_tasks {
    poll_single_local_task(
      app_handle,
      app_env_configs,
      app_data_root,
      task_database,
      storyteller_creds_manager,
      task,
    ).await;
  }
}

async fn poll_single_local_task(
  app_handle: &AppHandle,
  app_env_configs: &AppEnvConfigs,
  app_data_root: &AppDataRoot,
  task_database: &TaskDatabase,
  storyteller_creds_manager: &StorytellerCredentialManager,
  task: &Task,
) {
  let Some(status_url) = task.queue_status_url.as_deref() else {
    handle_local_failure(app_handle, task_database, task, "No local job status URL to check").await;
    return;
  };

  let status = match poll_local_image_job(status_url).await {
    Ok(status) => status,
    Err(LocalImageServerError::Unreachable(url)) => {
      // Keep the task pending: the server may be restarting or not started yet.
      warn!("[LocalPolling] Server unreachable for task {} ({}); will retry", task.id.as_str(), url);
      return;
    }
    Err(err) => {
      handle_local_failure(app_handle, task_database, task, &err.to_string()).await;
      return;
    }
  };

  match status.status {
    LocalJobState::Queued | LocalJobState::Running => {
      info!("[LocalPolling] Task {} still {:?}", task.id.as_str(), status.status);
    }
    LocalJobState::Failed => {
      let reason = status.error.unwrap_or_else(|| "Local generation failed".to_string());
      handle_local_failure(app_handle, task_database, task, &reason).await;
    }
    LocalJobState::Completed => {
      handle_local_complete(
        app_handle,
        app_env_configs,
        app_data_root,
        task_database,
        storyteller_creds_manager,
        task,
        &status.images,
      ).await;
    }
  }
}
