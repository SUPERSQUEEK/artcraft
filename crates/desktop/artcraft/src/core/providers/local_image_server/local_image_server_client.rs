//! Client for the on-device image inference server (eg. Qwen-Image on Apple Silicon).
//!
//! The server runs outside the app (see `local_image_server/` at the repo root) and
//! exposes a tiny job queue: enqueue a prompt, then poll the job until it finishes.
//! Finished images are served back over HTTP so the normal download/upload path works.

use log::info;
use serde_derive::{Deserialize, Serialize};

const DEFAULT_LOCAL_IMAGE_SERVER_URL: &str = "http://127.0.0.1:8189";
const LOCAL_IMAGE_SERVER_URL_ENV: &str = "ARTCRAFT_LOCAL_IMAGE_SERVER_URL";

// ── API types ──

#[derive(Serialize, Debug)]
pub struct LocalEnqueueRequest<'a> {
  pub model: &'a str,
  pub prompt: &'a str,
  /// Output size. Omit both to follow the last reference image's aspect ratio.
  #[serde(skip_serializing_if = "Option::is_none")]
  pub width: Option<u32>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub height: Option<u32>,
  pub num_images: u32,
  /// Reference images for editing: http(s) URLs or base64-encoded image bytes.
  #[serde(skip_serializing_if = "Vec::is_empty")]
  pub images: Vec<String>,
}

#[derive(Deserialize, Debug)]
pub struct LocalEnqueueResponse {
  pub job_id: String,
}

#[derive(Deserialize, Debug)]
pub struct LocalJobStatusResponse {
  pub status: LocalJobState,
  /// Absolute URLs of finished images (only when `Completed`).
  #[serde(default)]
  pub images: Vec<String>,
  pub error: Option<String>,
}

#[derive(Deserialize, Debug, Copy, Clone, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LocalJobState {
  Queued,
  Running,
  Completed,
  Failed,
}

#[derive(Debug)]
pub enum LocalImageServerError {
  /// The server isn't listening (not started, or crashed).
  Unreachable(String),
  /// The server doesn't know this job (eg. it restarted and lost its queue).
  JobNotFound,
  BadResponse(String),
}

impl std::fmt::Display for LocalImageServerError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Unreachable(url) => write!(f, "Local image server isn't running at {}. Start it with local_image_server/serve.sh", url),
      Self::JobNotFound => write!(f, "Local image server doesn't know this job (was it restarted?)"),
      Self::BadResponse(msg) => write!(f, "Local image server returned a bad response: {}", msg),
    }
  }
}

impl std::error::Error for LocalImageServerError {}

// ── Public API ──

pub fn local_image_server_url() -> String {
  std::env::var(LOCAL_IMAGE_SERVER_URL_ENV)
      .ok()
      .filter(|url| !url.trim().is_empty())
      .unwrap_or_else(|| DEFAULT_LOCAL_IMAGE_SERVER_URL.to_string())
      .trim_end_matches('/')
      .to_string()
}

pub fn local_job_status_url(job_id: &str) -> String {
  format!("{}/v1/jobs/{}", local_image_server_url(), job_id)
}

pub async fn enqueue_local_image_job(request: &LocalEnqueueRequest<'_>) -> Result<LocalEnqueueResponse, LocalImageServerError> {
  let base_url = local_image_server_url();
  let url = format!("{}/v1/generate", base_url);
  info!("Enqueueing local image job at {}: {:?}", url, request);

  let response = reqwest::Client::new()
      .post(&url)
      .json(request)
      .send()
      .await
      .map_err(|err| map_send_error(err, &base_url))?;

  parse_json(response).await
}

pub async fn poll_local_image_job(status_url: &str) -> Result<LocalJobStatusResponse, LocalImageServerError> {
  let response = reqwest::Client::new()
      .get(status_url)
      .send()
      .await
      .map_err(|err| map_send_error(err, status_url))?;

  if response.status() == reqwest::StatusCode::NOT_FOUND {
    return Err(LocalImageServerError::JobNotFound);
  }

  parse_json(response).await
}

// ── Helpers ──

async fn parse_json<T: serde::de::DeserializeOwned>(response: reqwest::Response) -> Result<T, LocalImageServerError> {
  let status = response.status();
  let body = response.text().await
      .map_err(|err| LocalImageServerError::BadResponse(err.to_string()))?;

  if !status.is_success() {
    return Err(LocalImageServerError::BadResponse(format!("HTTP {}: {}", status, body)));
  }

  serde_json::from_str(&body)
      .map_err(|err| LocalImageServerError::BadResponse(format!("{}: {}", err, body)))
}

fn map_send_error(err: reqwest::Error, url: &str) -> LocalImageServerError {
  if err.is_connect() || err.is_timeout() {
    LocalImageServerError::Unreachable(url.to_string())
  } else {
    LocalImageServerError::BadResponse(err.to_string())
  }
}
