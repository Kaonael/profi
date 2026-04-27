// SPDX-License-Identifier: Apache-2.0

//! End-to-end TLS test for the `/metrics` server.
//!
//! Spins up axum-server on an ephemeral port with a self-signed CA,
//! server cert, and two client certs (trusted / untrusted). Exercises:
//!
//! * `server` TLS mode — any client connects, auth=off → 200.
//! * `mtls` TLS mode — trusted client cert connects → 200; untrusted
//!   client cert fails the handshake.
//! * `/health` bypasses auth (even when auth_mode != off).
//!
//! TokenReview bearer validation is NOT covered here — it requires a
//! live Kubernetes API server. See unit tests in `http_security.rs` for
//! the validation / cache logic.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use profi::http_security::{AuthMode, MetricsSecurityArgs, TlsMode};
use profi::metrics::{KernelMode, Metrics};
use profi::server::{serve_http, AppState};
use rcgen::{CertificateParams, DistinguishedName, KeyPair, KeyUsagePurpose};
use tempfile::TempDir;

struct CertBundle {
    _dir: TempDir,
    ca_pem: String,
    server_cert_path: std::path::PathBuf,
    server_key_path: std::path::PathBuf,
    ca_path: std::path::PathBuf,
    client_cert_pem: String,
    client_key_pem: String,
    untrusted_client_cert_pem: String,
    untrusted_client_key_pem: String,
}

fn build_certs() -> Result<CertBundle> {
    let dir = tempfile::tempdir()?;

    let mut ca_params = CertificateParams::new(vec!["profi-test-ca".into()])?;
    ca_params.distinguished_name = {
        let mut dn = DistinguishedName::new();
        dn.push(rcgen::DnType::CommonName, "profi-test-ca");
        dn
    };
    ca_params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
    ca_params.key_usages = vec![
        KeyUsagePurpose::KeyCertSign,
        KeyUsagePurpose::CrlSign,
        KeyUsagePurpose::DigitalSignature,
    ];
    let ca_key = KeyPair::generate()?;
    let ca_cert = ca_params.self_signed(&ca_key)?;
    let ca_pem = ca_cert.pem();

    let mut srv_params = CertificateParams::new(vec!["localhost".into(), "127.0.0.1".into()])?;
    srv_params.distinguished_name = {
        let mut dn = DistinguishedName::new();
        dn.push(rcgen::DnType::CommonName, "profi-server");
        dn
    };
    srv_params.key_usages = vec![KeyUsagePurpose::DigitalSignature];
    srv_params.extended_key_usages = vec![rcgen::ExtendedKeyUsagePurpose::ServerAuth];
    let srv_key = KeyPair::generate()?;
    let srv_cert = srv_params.signed_by(&srv_key, &ca_cert, &ca_key)?;
    let server_chain_pem = format!("{}{}", srv_cert.pem(), ca_pem);
    let server_key_pem = srv_key.serialize_pem();

    let server_cert_path = dir.path().join("server.crt");
    let server_key_path = dir.path().join("server.key");
    let ca_path = dir.path().join("ca.crt");
    std::fs::write(&server_cert_path, &server_chain_pem)?;
    std::fs::write(&server_key_path, &server_key_pem)?;
    std::fs::write(&ca_path, &ca_pem)?;

    let mut cli_params = CertificateParams::new(vec!["profi-client".into()])?;
    cli_params.distinguished_name = {
        let mut dn = DistinguishedName::new();
        dn.push(rcgen::DnType::CommonName, "profi-client");
        dn
    };
    cli_params.extended_key_usages = vec![rcgen::ExtendedKeyUsagePurpose::ClientAuth];
    let cli_key = KeyPair::generate()?;
    let cli_cert = cli_params.signed_by(&cli_key, &ca_cert, &ca_key)?;
    let client_cert_pem = cli_cert.pem();
    let client_key_pem = cli_key.serialize_pem();

    let mut other_ca_params = CertificateParams::new(vec!["profi-other-ca".into()])?;
    other_ca_params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
    let other_ca_key = KeyPair::generate()?;
    let other_ca_cert = other_ca_params.self_signed(&other_ca_key)?;

    let mut ucli_params = CertificateParams::new(vec!["profi-untrusted".into()])?;
    ucli_params.extended_key_usages = vec![rcgen::ExtendedKeyUsagePurpose::ClientAuth];
    let ucli_key = KeyPair::generate()?;
    let ucli_cert = ucli_params.signed_by(&ucli_key, &other_ca_cert, &other_ca_key)?;
    let untrusted_client_cert_pem = ucli_cert.pem();
    let untrusted_client_key_pem = ucli_key.serialize_pem();

    Ok(CertBundle {
        _dir: dir,
        ca_pem,
        server_cert_path,
        server_key_path,
        ca_path,
        client_cert_pem,
        client_key_pem,
        untrusted_client_cert_pem,
        untrusted_client_key_pem,
    })
}

fn make_state(metrics: Metrics) -> Arc<AppState> {
    Arc::new(AppState {
        metrics,
        heartbeat_ns: AtomicU64::new(0),
        ring_buffer_open: AtomicBool::new(true),
        libs_attached: AtomicUsize::new(1),
        k8s_ready: AtomicBool::new(true),
        events_processed: AtomicU64::new(0),
        kernel_mode: "anonymous".into(),
        start_time: Instant::now(),
    })
}

fn free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

async fn wait_for_port(addr: SocketAddr) {
    for _ in 0..100 {
        if std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(50)).is_ok() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("server did not start on {addr}");
}

#[tokio::test]
async fn tls_server_mode_allows_any_client() -> Result<()> {
    let _ = rustls::crypto::ring::default_provider().install_default();

    let bundle = build_certs()?;
    let metrics = Metrics::new(KernelMode::Off)?;
    let state = make_state(metrics);

    let port = free_port();
    let listen = format!("127.0.0.1:{port}");

    let security = MetricsSecurityArgs {
        metrics_tls_mode: TlsMode::Server,
        metrics_tls_cert: Some(bundle.server_cert_path.clone()),
        metrics_tls_key: Some(bundle.server_key_path.clone()),
        metrics_tls_client_ca: None,
        metrics_auth_mode: AuthMode::Off,
        metrics_auth_audience: None,
        metrics_auth_cache_ttl: 60,
        metrics_auth_cache_size: 1024,
    };
    security.validate()?;

    let listen_c = listen.clone();
    let handle = tokio::spawn(async move {
        let _ = serve_http(listen_c, state, security, None).await;
    });

    let addr: SocketAddr = listen.parse()?;
    wait_for_port(addr).await;

    let client = reqwest::Client::builder()
        .add_root_certificate(reqwest::Certificate::from_pem(bundle.ca_pem.as_bytes())?)
        .build()?;
    let url = format!("https://localhost:{port}/metrics");
    let resp = client.get(&url).send().await?;
    assert_eq!(resp.status(), 200);
    let body = resp.text().await?;
    assert!(body.contains("profi_"));

    handle.abort();
    Ok(())
}

#[tokio::test]
async fn mtls_trusted_client_succeeds_untrusted_fails() -> Result<()> {
    let _ = rustls::crypto::ring::default_provider().install_default();

    let bundle = build_certs()?;
    let metrics = Metrics::new(KernelMode::Off)?;
    let state = make_state(metrics);

    let port = free_port();
    let listen = format!("127.0.0.1:{port}");

    let security = MetricsSecurityArgs {
        metrics_tls_mode: TlsMode::Mtls,
        metrics_tls_cert: Some(bundle.server_cert_path.clone()),
        metrics_tls_key: Some(bundle.server_key_path.clone()),
        metrics_tls_client_ca: Some(bundle.ca_path.clone()),
        metrics_auth_mode: AuthMode::Off,
        metrics_auth_audience: None,
        metrics_auth_cache_ttl: 60,
        metrics_auth_cache_size: 1024,
    };
    security.validate()?;

    let listen_c = listen.clone();
    let handle = tokio::spawn(async move {
        let _ = serve_http(listen_c, state, security, None).await;
    });

    let addr: SocketAddr = listen.parse()?;
    wait_for_port(addr).await;

    let ca = reqwest::Certificate::from_pem(bundle.ca_pem.as_bytes())?;

    let trusted_pem = format!("{}{}", bundle.client_cert_pem, bundle.client_key_pem);
    let identity = reqwest::Identity::from_pem(trusted_pem.as_bytes())?;
    let trusted_client = reqwest::Client::builder()
        .add_root_certificate(ca.clone())
        .identity(identity)
        .build()?;
    let url = format!("https://localhost:{port}/metrics");
    let resp = trusted_client.get(&url).send().await?;
    assert_eq!(resp.status(), 200, "trusted client should succeed");

    let untrusted_pem = format!(
        "{}{}",
        bundle.untrusted_client_cert_pem, bundle.untrusted_client_key_pem
    );
    let u_identity = reqwest::Identity::from_pem(untrusted_pem.as_bytes())?;
    let untrusted_client = reqwest::Client::builder()
        .add_root_certificate(ca)
        .identity(u_identity)
        .build()?;

    let untrusted_result = untrusted_client.get(&url).send().await;
    assert!(
        untrusted_result.is_err(),
        "untrusted client should fail the TLS handshake, got: {untrusted_result:?}"
    );

    handle.abort();
    Ok(())
}

#[tokio::test]
async fn auth_mode_rejects_unauthenticated_metrics_but_allows_health() -> Result<()> {
    let _ = rustls::crypto::ring::default_provider().install_default();

    let bundle = build_certs()?;
    let metrics = Metrics::new(KernelMode::Off)?;
    let state = make_state(metrics.clone());

    let port = free_port();
    let listen = format!("127.0.0.1:{port}");

    let security = MetricsSecurityArgs {
        metrics_tls_mode: TlsMode::Server,
        metrics_tls_cert: Some(bundle.server_cert_path.clone()),
        metrics_tls_key: Some(bundle.server_key_path.clone()),
        metrics_tls_client_ca: None,
        metrics_auth_mode: AuthMode::Bearer,
        metrics_auth_audience: None,
        metrics_auth_cache_ttl: 60,
        metrics_auth_cache_size: 1024,
    };
    security.validate()?;

    let listen_c = listen.clone();
    let handle = tokio::spawn(async move {
        let _ = serve_http(listen_c, state, security, None).await;
    });

    let addr: SocketAddr = listen.parse()?;
    wait_for_port(addr).await;

    let client = reqwest::Client::builder()
        .add_root_certificate(reqwest::Certificate::from_pem(bundle.ca_pem.as_bytes())?)
        .build()?;

    let health_url = format!("https://localhost:{port}/health");
    let resp = client.get(&health_url).send().await?;
    assert_eq!(resp.status(), 200, "/health must bypass auth");

    let metrics_url = format!("https://localhost:{port}/metrics");
    let resp = client.get(&metrics_url).send().await?;
    assert_eq!(resp.status(), 401, "/metrics without auth must 401");

    let auth_fail_count = metrics
        .http_auth_failures
        .with_label_values(&["no_auth"])
        .get();
    assert!(auth_fail_count >= 1);

    handle.abort();
    Ok(())
}
