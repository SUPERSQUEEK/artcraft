use crate::core::api_adapters::models::image::tauri_image_model_to_generation_model::tauri_image_model_to_generation_model;
use crate::core::commands::enqueue::generate_error::{BadInputReason, GenerateError};
use crate::core::commands::enqueue::task_enqueue_success::TaskEnqueueSuccess;
use crate::core::commands::generate::generate_image::tauri_generate_image_request::TauriGenerateImageRequest;
use crate::core::commands::generate::generate_image::tauri_image_model::TauriImageModel;
use crate::core::providers::local_image_server::local_image_server_client::{
  enqueue_local_image_job, local_job_status_url, LocalEnqueueRequest,
};
use crate::core::state::app_env_configs::app_env_configs::AppEnvConfigs;
use crate::services::storyteller::state::storyteller_credential_manager::StorytellerCredentialManager;
use anyhow::anyhow;
use artcraft_api_defs::prompts::create_prompt::CreatePromptRequest;
use artcraft_client::endpoints::prompts::create_prompt::create_prompt;
use enums::common::generation::common_aspect_ratio::CommonAspectRatio;
use enums::common::generation::common_model_type::CommonModelType;
use enums::common::generation_provider::GenerationProvider;
use enums::tauri::tasks::task_type::TaskType;
use log::warn;
use tokens::tokens::prompts::PromptToken;
use uuid_utils::uuid::generate_random_uuid;

const MAX_BATCH_SIZE: u32 = 4;

/// Enqueue a text-to-image job on the on-device inference server.
/// The third-party polling thread picks the task up and finishes it.
pub async fn handle_local(
  request: &TauriGenerateImageRequest,
  config: &AppEnvConfigs,
  storyteller: &StorytellerCredentialManager,
) -> Result<TaskEnqueueSuccess, GenerateError> {
  let model = request.model.ok_or_else(GenerateError::no_model_specified)?;
  let server_model = match model {
    TauriImageModel::QwenImage2p1TurboLocal => "qwen_image_2p1_turbo",
    _ => return Err(bad_input("This model can't run on the local image server")),
  };
  if has_image_inputs(request) {
    return Err(bad_input("The local model doesn't support image references yet"));
  }
  let prompt = request.prompt.as_deref()
      .map(str::trim)
      .filter(|prompt| !prompt.is_empty())
      .ok_or_else(|| bad_input("A prompt is required"))?;

  let (width, height) = dimensions_for_aspect_ratio(request.aspect_ratio);
  let num_images = request.batch_size.unwrap_or(1).clamp(1, MAX_BATCH_SIZE);

  let maybe_prompt_token = create_prompt_record(config, storyteller, prompt, request.aspect_ratio, num_images).await;

  let job = enqueue_local_image_job(&LocalEnqueueRequest {
    model: server_model,
    prompt,
    width,
    height,
    num_images,
  }).await.map_err(|err| GenerateError::AnyhowError(anyhow!(err.to_string())))?;

  Ok(TaskEnqueueSuccess {
    task_type: TaskType::ImageGeneration,
    model: Some(tauri_image_model_to_generation_model(model)),
    provider: GenerationProvider::Local,
    maybe_queue_status_url: Some(local_job_status_url(&job.job_id)),
    provider_job_id: Some(job.job_id),
    maybe_queue_response_url: None,
    maybe_prompt_token,
  })
}

/// Store the prompt so library entries show it. Fails open: a missing prompt
/// record shouldn't block an on-device generation.
async fn create_prompt_record(
  config: &AppEnvConfigs,
  storyteller: &StorytellerCredentialManager,
  prompt: &str,
  aspect_ratio: Option<CommonAspectRatio>,
  num_images: u32,
) -> Option<PromptToken> {
  let creds = match storyteller.get_credentials() {
    Ok(Some(creds)) => creds,
    _ => {
      warn!("[Local] No ArtCraft credentials; skipping prompt record");
      return None;
    }
  };
  let request = CreatePromptRequest {
    uuid_idempotency_token: generate_random_uuid(),
    positive_prompt: Some(prompt.to_string()),
    negative_prompt: None,
    model_type: Some(CommonModelType::Qwen),
    // NB: The backend doesn't know the "local" provider, so leave it unset.
    generation_provider: None,
    maybe_generation_mode: None,
    maybe_aspect_ratio: aspect_ratio,
    maybe_resolution: None,
    maybe_batch_count: Some(num_images as u8),
    maybe_generate_audio: None,
    maybe_duration_seconds: None,
  };
  match create_prompt(&config.storyteller_host, Some(&creds), request).await {
    Ok(response) => Some(response.prompt_token),
    Err(err) => {
      warn!("[Local] Failed to create prompt record (failing open): {:?}", err);
      None
    }
  }
}

/// ~1 megapixel sizes; both sides are multiples of 16 as the model requires.
fn dimensions_for_aspect_ratio(aspect_ratio: Option<CommonAspectRatio>) -> (u32, u32) {
  match aspect_ratio {
    Some(CommonAspectRatio::WideThreeByTwo) => (1248, 832),
    Some(CommonAspectRatio::WideFourByThree) => (1184, 880),
    Some(CommonAspectRatio::WideFiveByFour) => (1152, 928),
    Some(CommonAspectRatio::WideSixteenByNine | CommonAspectRatio::Wide) => (1360, 768),
    Some(CommonAspectRatio::WideTwentyOneByNine) => (1568, 672),
    Some(CommonAspectRatio::TallTwoByThree) => (832, 1248),
    Some(CommonAspectRatio::TallThreeByFour) => (880, 1184),
    Some(CommonAspectRatio::TallFourByFive) => (928, 1152),
    Some(CommonAspectRatio::TallNineBySixteen | CommonAspectRatio::Tall) => (768, 1360),
    Some(CommonAspectRatio::TallNineByTwentyOne) => (672, 1568),
    _ => (1024, 1024),
  }
}

fn has_image_inputs(request: &TauriGenerateImageRequest) -> bool {
  request.image_media_tokens.as_ref().is_some_and(|images| !images.is_empty())
      || request.canvas_image_media_token.is_some()
      || request.canvas_image_raw_bytes.is_some()
      || request.scene_image_media_token.is_some()
      || request.scene_image_raw_bytes.is_some()
      || request.inpainting_mask_image_media_token.is_some()
      || request.inpainting_mask_image_raw_bytes.is_some()
}

fn bad_input(message: &str) -> GenerateError {
  GenerateError::BadInput(BadInputReason::WrongImageArguments(message.to_string()))
}
