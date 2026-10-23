// Download utilities — a MoonBit port of `reasoning_from_scratch/utils.py`
// by way of the Rust port in `learn_reasoning/src/util.rs`.
//
// Downloads go to the same cache directory as the Rust and Elixir ports
// (`~/.cache/models/Qwen/Qwen3-0.6B`) so the multi-GB checkpoint is fetched
// once and shared by all implementations.

///|
/// Errors raised by the download helpers in this file.
pub(all) suberror DownloadError {
  /// Server answered with a non-success HTTP status (status, url).
  BadStatus(Int32, String)
  /// Redirect chain exceeded the hop limit.
  TooManyRedirects(String)
  /// Primary (and backup, when given) URL both failed.
  DownloadFailed(String)
  /// Redirect `Location` was neither absolute nor origin-relative.
  BadRedirect(String)
}

///|
/// `~/.cache/models/Qwen/Qwen3-0.6B` — the shared model cache directory
/// (same location as the Rust `qwen3::model_dir()`).
pub fn qwen3_model_dir() -> String {
  let home = @env.get_env_var("HOME").unwrap_or(".")
  "\{home}/.cache/models/Qwen/Qwen3-0.6B"
}

///|
/// Last path segment of `url`, with any query string or fragment stripped.
/// (Port of `filename_from_url` in util.rs.)
fn filename_from_url(url : String) -> String {
  let qm = url.find("?")
  let hash = url.find("#")
  let end = match (qm, hash) {
    (Some(a), Some(b)) => if a < b { a } else { b }
    (Some(a), None) => a
    (None, Some(b)) => b
    (None, None) => url.length()
  }
  let path = url[:end]
  let start = match path.rev_find("/") {
    Some(i) => i + 1
    None => 0
  }
  path[start:].to_owned()
}

///|
/// HTTP(S) proxy URL from the standard environment variables, if any.
/// (reqwest did this implicitly in the Rust port; here it is explicit.)
fn proxy_url_from_env() -> String? {
  for name in ["https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY"] {
    if @env.get_env_var(name) is Some(v) && v != "" {
      return Some(v)
    }
  }
  None
}

///|
/// Resolve a redirect `Location` against the URL that produced it.
/// HF sends absolute URLs for CDN files but origin-relative paths
/// ("/api/resolve-cache/...") for small files like README.md.
fn resolve_redirect(current : String, location : String) -> String raise {
  if location.has_prefix("http") {
    return location
  }
  if location.has_prefix("/") {
    let after_scheme = match current.find("://") {
      Some(i) => i + 3
      None => raise DownloadError::BadRedirect(location)
    }
    let host_end = match current[after_scheme:].find("/") {
      Some(i) => after_scheme + i
      None => current.length()
    }
    return current[:host_end].to_owned() + location
  }
  raise DownloadError::BadRedirect(location)
}

///|
/// Open a streaming GET, following redirects manually (the async HTTP client
/// does not follow them on its own, and huggingface.co redirects to a CDN).
///
/// A fresh proxy client is built per hop: the HTTP client takes ownership of
/// the proxy client as its transport, so a proxy client cannot be reused
/// across requests.
async fn open_stream(
  url : String,
  proxy_url : String?,
) -> (@http.Response, @http.Client) {
  let mut url = url
  for _ in 0..<5 {
    let proxy = match proxy_url {
      Some(p) => Some(@http.Client(p))
      None => None
    }
    let (resp, body) = @http.get_stream(url, proxy?)
    if resp.code is (301 | 302 | 303 | 307 | 308) {
      match resp.headers.get(@http.CaseInsensitiveString("location")) {
        Some(location) => {
          body.close()
          url = resolve_redirect(url, location)
          continue
        }
        None => raise DownloadError::BadStatus(resp.code, url)
      }
    }
    if resp.code < 200 || resp.code >= 300 {
      body.close()
      raise DownloadError::BadStatus(resp.code, url)
    }
    return (resp, body)
  }
  raise DownloadError::TooManyRedirects(url)
}

///|
/// Download `url` to `dest`, skipping when the local file already matches
/// the remote Content-Length, and showing progress while streaming.
async fn try_download(
  url : String,
  dest : String,
  filename : String,
  proxy_url : String?,
) -> Unit {
  let (resp, body) = open_stream(url, proxy_url)
  defer body.close()
  let remote_size : Int64? = match
    resp.headers.get(@http.CaseInsensitiveString("content-length")) {
    Some(s) => Some(@string.parse_int64(s)) catch { _ => None }
    None => None
  }
  // Skip download if the local file already matches the remote size.
  if remote_size is Some(size) &&
    @fs.exists(dest) &&
    @fs.file_size(dest) == size {
    println("✓ \{dest} already up-to-date")
    return
  }
  let file = @fs.open(dest, mode=WriteOnly, create_mode=CreateOrTruncate)
  defer file.close()
  let mut downloaded : Int64 = 0
  while true {
    match body.read_some(max_len=65536) {
      Some(chunk) => {
        file.write(chunk)
        downloaded += chunk.length().to_int64()
        if remote_size is Some(size) && size > 0 {
          let pct = downloaded * 100 / size
          @stdio.stdout.write(
            "\r\{filename}: \{pct}% (\{downloaded / 1048576} MiB / \{size / 1048576} MiB)",
          )
        }
      }
      None => break
    }
  }
  if remote_size is Some(_) {
    println("")
  }
}

///|
/// Download `url` into `out_dir`, skipping files that are already complete.
/// Returns the destination path. On failure of the primary URL, falls back
/// to `backup_url` when provided. (Port of `download_file` in util.rs.)
pub async fn download_file(
  url : String,
  out_dir : String,
  backup_url? : String,
) -> String {
  @fs.mkdir(out_dir, recursive=true, allow_exist=true)
  let filename = filename_from_url(url)
  let dest = "\{out_dir}/\{filename}"
  let proxy_url = proxy_url_from_env()
  let primary_ok = try {
    try_download(url, dest, filename, proxy_url)
    true
  } catch {
    _ => false
  }
  if primary_ok {
    return dest
  }
  if backup_url is Some(backup) {
    println("Primary URL (\{url}) failed.\nTrying backup URL (\{backup})...")
    let backup_ok = try {
      try_download(backup, dest, filename, proxy_url)
      true
    } catch {
      _ => false
    }
    if backup_ok {
      return dest
    }
    raise DownloadError::DownloadFailed(
      "Failed to download \{filename}.\nPrimary URL failed (\{url})\nBackup URL failed (\{backup})",
    )
  }
  raise DownloadError::DownloadFailed(
    "Failed to download \{filename}.\nPrimary URL failed (\{url})",
  )
}

///|
test "filename_from_url" {
  assert_eq(
    filename_from_url(
      "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors",
    ),
    "model.safetensors",
  )
  assert_eq(
    filename_from_url("https://example.com/a/b/file.json?download=true"),
    "file.json",
  )
  assert_eq(
    filename_from_url("https://example.com/a/b/file.json#frag"),
    "file.json",
  )
  assert_eq(filename_from_url("file.bin"), "file.bin")
}

///|
test "qwen3_model_dir shares the cache with the Rust port" {
  assert_true(qwen3_model_dir().has_suffix("/.cache/models/Qwen/Qwen3-0.6B"))
}
