use crate::core::state::app_env_configs::app_env_configs::AppEnvConfigs;
use crate::core::state::data_dir::app_data_root::AppDataRoot;
use crate::core::state::data_dir::trait_data_subdir::DataSubdir;
use crate::core::state::task_database::TaskDatabase;
use crate::core::threads::third_party_task_polling_thread::events::notify_frontend_of_completion::{
  notify_frontend_of_completion, CompletionData,
};
use crate::core::threads::third_party_task_polling_thread::handlers::local::handle_local_failure::handle_local_failure;
use crate::core::utils::auto_download::{auto_download_task_urls, clear_auto_download_checkpoint};
use crate::services::storyteller::state::storyteller_credential_manager::StorytellerCredentialManager;
use artcraft_api_defs::utils::media_links_to_thumbnail_template::media_links_to_thumbnail_template;
use artcraft_client::endpoints::media_files::get_media_file::get_media_file;
use artcraft_client::endpoints::media_files::upload_image_media_file_from_file::{
  upload_image_media_file_from_file, UploadImageFromFileArgs,
};
use enums::tauri::tasks::task_media_file_class::TaskMediaFileClass;
use log::{error, info, warn};
use sqlite_tasks::queries::task::Task;
use sqlite_tasks::queries::update_successful_task_status_with_metadata::{
  update_successful_task_status_with_metadata, UpdateSuccessfulTaskArgs,
};
use std::path::PathBuf;
use tauri::AppHandle;
use tokens::tokens::batch_generations::BatchGenerationToken;
use tokens::tokens::media_files::MediaFileToken;
use tokens::tokens::prompts::PromptToken;
use uuid_utils::uuid::generate_random_uuid;

/// Download finished images from the local server, upload them to the user's
/// library, and tell the frontend — the same end state as a cloud generation.
pub async fn handle_local_complete(
  app_handle: &AppHandle,
  app_env_configs: &AppEnvConfigs,
  app_data_root: &AppDataRoot,
  task_database: &TaskDatabase,
  storyteller_creds_manager: &StorytellerCredentialManager,
  task: &Task,
  image_urls: &[String],
) {
  info!("[LocalComplete] Handling completed task {}", task.id.as_str());

  if image_urls.is_empty() {
    handle_local_failure(app_handle, task_database, task, "Local job finished without images").await;
    return;
  }

  let result = handle_local_complete_inner(
    app_handle,
    app_env_configs,
    app_data_root,
    task_database,
    storyteller_creds_manager,
    task,
    image_urls,
  ).await;

  if let Err(err) = result {
    // Leave the task pending so the next poll retries (eg. transient upload errors).
    error!("[LocalComplete] Failed to handle task {}: {:?}", task.id.as_str(), err);
  }
}

async fn handle_local_complete_inner(
  app_handle: &AppHandle,
  app_env_configs: &AppEnvConfigs,
  app_data_root: &AppDataRoot,
  task_database: &TaskDatabase,
  storyteller_creds_manager: &StorytellerCredentialManager,
  task: &Task,
  image_urls: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
  let creds = storyteller_creds_manager.get_credentials()?
      .ok_or("No ArtCraft credentials available to upload local results")?;

  // Save local copies (per the user's auto-download preference) before uploading.
  let download_urls = image_urls.iter().map(|url| reqwest::Url::parse(url)).collect::<Result<Vec<_>, _>>()?;
  auto_download_task_urls(app_handle, task, &download_urls).await?;

  let maybe_prompt_token = task.prompt_token.as_ref().map(|s| PromptToken::new_from_str(s));
  let maybe_batch_token = (image_urls.len() > 1).then(BatchGenerationToken::generate);

  let mut maybe_primary_media_file_token: Option<MediaFileToken> = None;

  for (i, url) in image_urls.iter().enumerate() {
    let path = download_to_temp(url, app_data_root, i).await?;
    let upload = upload_image_media_file_from_file(UploadImageFromFileArgs {
      api_host: &app_env_configs.storyteller_host,
      maybe_creds: Some(&creds),
      path: &path,
      is_intermediate_system_file: false,
      maybe_prompt_token: maybe_prompt_token.as_ref(),
      maybe_batch_token: maybe_batch_token.as_ref(),
      // NB: The backend doesn't know the "local" provider.
      maybe_generation_provider: None,
    }).await?;
    let _ = std::fs::remove_file(&path);
    info!("[LocalComplete] Uploaded result {} as {:?}", i, upload.media_file_token);
    maybe_primary_media_file_token.get_or_insert(upload.media_file_token);
  }

  let mut maybe_cdn_url: Option<reqwest::Url> = None;
  let mut maybe_thumbnail_url_template = None;

  if let Some(media_file_token) = maybe_primary_media_file_token.as_ref() {
    match get_media_file(&app_env_configs.storyteller_host, media_file_token).await {
      Ok(response) => {
        maybe_cdn_url = Some(response.media_file.media_links.cdn_url.clone());
        maybe_thumbnail_url_template = media_links_to_thumbnail_template(&response.media_file.media_links)
            .map(|s| s.to_string());
      }
      Err(err) => warn!("[LocalComplete] Failed to look up media file after upload: {:?} (failing open)", err),
    }
  }

  let updated = update_successful_task_status_with_metadata(UpdateSuccessfulTaskArgs {
    db: task_database.get_connection(),
    task_id: &task.id,
    maybe_batch_token: maybe_batch_token.as_ref(),
    maybe_primary_media_file_token: maybe_primary_media_file_token.as_ref(),
    maybe_primary_media_file_class: Some(TaskMediaFileClass::Image),
    maybe_primary_media_file_cdn_url: maybe_cdn_url.as_ref().map(|url| url.as_str()),
    maybe_primary_media_file_thumbnail_url_template: maybe_thumbnail_url_template.as_deref(),
  }).await?;

  if updated {
    clear_auto_download_checkpoint(app_handle, task);
    if let Some(primary_media_file_token) = maybe_primary_media_file_token {
      let completion = CompletionData {
        primary_media_file_token,
        maybe_cdn_url,
        maybe_thumbnail_url_template,
        maybe_batch_token,
        media_class: TaskMediaFileClass::Image,
      };
      notify_frontend_of_completion(
        app_handle,
        &app_env_configs.storyteller_host,
        Some(&creds),
        task,
        &completion,
      ).await;
    }
  }

  info!("[LocalComplete] Task {} fully handled", task.id.as_str());
  Ok(())
}

async fn download_to_temp(
  url: &str,
  app_data_root: &AppDataRoot,
  index: usize,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
  let bytes = reqwest::get(url).await?.error_for_status()?.bytes().await?;
  let path = app_data_root.temp_dir().path().join(format!("local_{}_{}.png", generate_random_uuid(), index));
  std::fs::write(&path, &bytes)?;
  Ok(path)
}
