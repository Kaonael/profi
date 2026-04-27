// SPDX-License-Identifier: Apache-2.0

use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::get;
use axum::Router;
use log::info;
use serde::Serialize;

use crate::http_security::{
    auth_middleware, build_rustls_config, AuthLayerState, AuthMode, MetricsSecurityArgs,
    TokenReviewer,
};
use crate::metrics::Metrics;

/// Shared application state for HTTP endpoints and the event loop.
pub struct AppState {
    pub metrics: Metrics,
    /// Epoch-like counter updated each event loop iteration (nanos from start_time).
    pub heartbeat_ns: AtomicU64,
    /// True once the ring buffer has been opened.
    pub ring_buffer_open: AtomicBool,
    /// Number of libraries successfully attached.
    pub libs_attached: AtomicUsize,
    /// True once K8s watcher is initialized (or NODE_NAME is unset).
    pub k8s_ready: AtomicBool,
    /// Total events processed.
    pub events_processed: AtomicU64,
    /// Kernel tracing mode (immutable after init).
    pub kernel_mode: String,
    /// Process start time.
    pub start_time: Instant,
}

const HEARTBEAT_STALE_NS: u64 = 30_000_000_000; // 30 seconds in nanoseconds

async fn handle_metrics(State(state): State<Arc<AppState>>) -> axum::response::Response {
    let body = state.metrics.encode_bytes();
    axum::response::Response::builder()
        .header(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )
        .body(axum::body::Body::from(body))
        .unwrap()
}

/// Liveness probe: 200 if event loop heartbeat is fresh (< 30s), else 503.
async fn handle_health(State(state): State<Arc<AppState>>) -> (StatusCode, &'static str) {
    let heartbeat = state.heartbeat_ns.load(Ordering::Relaxed);
    let now = state.start_time.elapsed().as_nanos() as u64;
    if heartbeat == 0 || now.saturating_sub(heartbeat) < HEARTBEAT_STALE_NS {
        (StatusCode::OK, "ok\n")
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "event loop stale\n")
    }
}

/// Readiness probe: 200 if at least 1 library attached, ring buffer open, and K8s ready.
async fn handle_ready(State(state): State<Arc<AppState>>) -> (StatusCode, &'static str) {
    let libs = state.libs_attached.load(Ordering::Relaxed);
    let rb_open = state.ring_buffer_open.load(Ordering::Relaxed);
    let k8s = state.k8s_ready.load(Ordering::Relaxed);

    if libs >= 1 && rb_open && k8s {
        (StatusCode::OK, "ready\n")
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "not ready\n")
    }
}

#[derive(Serialize)]
struct StatusResponse {
    attached_libraries: usize,
    tracked_pids: i64,
    events_processed: u64,
    uptime_seconds: f64,
    kernel_mode: String,
}

/// JSON status endpoint with operational details.
async fn handle_status(State(state): State<Arc<AppState>>) -> axum::Json<StatusResponse> {
    axum::Json(StatusResponse {
        attached_libraries: state.libs_attached.load(Ordering::Relaxed),
        tracked_pids: state.metrics.tracked_pids.get(),
        events_processed: state.events_processed.load(Ordering::Relaxed),
        uptime_seconds: state.start_time.elapsed().as_secs_f64(),
        kernel_mode: state.kernel_mode.clone(),
    })
}

fn base_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/metrics", get(handle_metrics))
        .route("/health", get(handle_health))
        .route("/ready", get(handle_ready))
        .route("/status", get(handle_status))
        .with_state(state)
}

/// Start the HTTP(S) server. When `security.tls_enabled()` is false this
/// keeps the historic plain-HTTP behaviour.
pub async fn serve_http(
    listen: String,
    state: Arc<AppState>,
    security: MetricsSecurityArgs,
    reviewer: Option<Arc<TokenReviewer>>,
) -> Result<()> {
    let metrics = state.metrics.clone();
    let auth_state = Arc::new(AuthLayerState {
        mode: security.metrics_auth_mode,
        mtls_listener: security.metrics_tls_mode == crate::http_security::TlsMode::Mtls,
        reviewer,
        metrics: metrics.clone(),
    });

    let app = if security.metrics_auth_mode == AuthMode::Off {
        base_router(state)
    } else {
        base_router(state).layer(axum::middleware::from_fn_with_state(
            auth_state,
            auth_middleware,
        ))
    };

    let addr: SocketAddr = listen.parse().context("parse --listen address")?;

    match build_rustls_config(&security)? {
        None => {
            let listener = tokio::net::TcpListener::bind(&addr).await?;
            info!("Prometheus endpoint: http://{addr}/metrics (plain HTTP)");
            axum::serve(listener, app).await?;
        }
        Some(tls_cfg) => {
            let scheme = if security.metrics_tls_mode == crate::http_security::TlsMode::Mtls {
                "https (mTLS)"
            } else {
                "https"
            };
            info!("Prometheus endpoint: {scheme}://{addr}/metrics");
            metrics
                .http_tls_handshakes
                .with_label_values(&["ok"])
                .inc_by(0); // eagerly materialise the label tuple
            axum_server::bind_rustls(
                addr,
                axum_server::tls_rustls::RustlsConfig::from_config(tls_cfg),
            )
            .serve(app.into_make_service())
            .await?;
        }
    }
    Ok(())
}
