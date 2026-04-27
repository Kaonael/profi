// SPDX-License-Identifier: Apache-2.0

//! Integration test: OTLP bridge pushes Prometheus registry contents over
//! gRPC to a mock MetricsServiceServer and we inspect the received proto.
//!
//! What we prove:
//!  - profi_* counter/gauge/histogram families reach the collector
//!  - Prometheus label `pod` is renamed to OTel semconv `k8s.pod.name`
//!  - Histograms carry N bucket_counts / N-1 explicit_bounds (the bridge
//!    arithmetic is one of the most bug-prone bits in any OTLP adapter)
//!  - `profi_system_*` self-observability metrics are suppressed on the wire

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use opentelemetry_proto::tonic::collector::metrics::v1::{
    metrics_service_server::{MetricsService, MetricsServiceServer},
    ExportMetricsServiceRequest, ExportMetricsServiceResponse,
};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tonic::{transport::Server, Request, Response, Status};

use profi::metrics::{KernelMode, Metrics};
use profi::otlp::{OtlpArgs, OtlpBridge, OtlpConfig};

#[derive(Clone, Default)]
struct MockCollector {
    received: Arc<Mutex<Vec<ExportMetricsServiceRequest>>>,
    notify: Arc<Mutex<Option<oneshot::Sender<()>>>>,
}

#[async_trait]
impl MetricsService for MockCollector {
    async fn export(
        &self,
        request: Request<ExportMetricsServiceRequest>,
    ) -> Result<Response<ExportMetricsServiceResponse>, Status> {
        self.received.lock().unwrap().push(request.into_inner());
        if let Some(tx) = self.notify.lock().unwrap().take() {
            let _ = tx.send(());
        }
        Ok(Response::new(ExportMetricsServiceResponse::default()))
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn otlp_bridge_exports_prometheus_metrics_to_mock_collector() {
    // Bind a random port so tests can run in parallel without collisions.
    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    drop(listener); // Free the port; tonic will rebind.

    let (export_tx, export_rx) = oneshot::channel();
    let collector = MockCollector {
        received: Arc::new(Mutex::new(Vec::new())),
        notify: Arc::new(Mutex::new(Some(export_tx))),
    };
    let received = collector.received.clone();

    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let server_addr = addr;
    let server_handle = tokio::spawn(async move {
        Server::builder()
            .add_service(MetricsServiceServer::new(collector))
            .serve_with_shutdown(server_addr, async {
                let _ = shutdown_rx.await;
            })
            .await
            .unwrap();
    });

    // Give the server a beat to bind before the bridge connects.
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Build a real profi Metrics and populate a couple of series.
    // cuda_calls labels (9): operation, pid, comm, namespace, pod, container, gpu, gpu_uuid, stream
    // cuda_duration labels (4): operation, namespace, pod, gpu
    let metrics = Metrics::new(KernelMode::Anonymous).expect("metrics");
    metrics
        .cuda_calls
        .with_label_values(&[
            "cudaLaunchKernel",
            "42",
            "python",
            "default",
            "demo-pod",
            "sglang",
            "0",
            "GPU-abcd",
            "default",
        ])
        .inc_by(5.0);
    metrics
        .cuda_duration
        .with_label_values(&["cudaLaunchKernel", "default", "demo-pod", "0"])
        .observe(0.005);
    metrics
        .cuda_duration
        .with_label_values(&["cudaLaunchKernel", "default", "demo-pod", "0"])
        .observe(0.02);
    // Self-observability metric — must be filtered by the `profi_system_` prefix rule.
    metrics.encode_duration.observe(0.0001);

    let otlp_args = OtlpArgs {
        otlp_endpoint: Some(format!("http://{}", addr)),
        otlp_protocol: "grpc".to_string(),
        otlp_interval_secs: 1, // fast tick for test
        otlp_timeout_secs: 5,
        otlp_service_name: "profi-test".to_string(),
        otlp_headers: None,
        otlp_ca_cert: None,
        otlp_client_cert: None,
        otlp_client_key: None,
        otlp_insecure: true,
        otlp_resource_attrs: None,
    };
    let cfg = OtlpConfig::resolve(&otlp_args)
        .expect("resolve")
        .expect("endpoint set");

    let _bridge = OtlpBridge::start(cfg, metrics, std::time::Instant::now()).expect("bridge start");

    // First export arrives after one full interval (bridge skips the initial
    // immediate tick). 5s is comfortable headroom over 1s.
    tokio::time::timeout(Duration::from_secs(5), export_rx)
        .await
        .expect("export within deadline")
        .expect("sender alive");

    {
        // Collect and verify.
        let requests = received.lock().unwrap();
        assert!(!requests.is_empty(), "at least one export received");
        let req = &requests[0];
        assert_eq!(req.resource_metrics.len(), 1);
        let rm = &req.resource_metrics[0];
        let scope_metrics = &rm.scope_metrics;
        assert!(
            !scope_metrics.is_empty(),
            "scope_metrics present, got {:?}",
            scope_metrics
        );
        let metrics_proto = &scope_metrics[0].metrics;

        let names: Vec<&str> = metrics_proto.iter().map(|m| m.name.as_str()).collect();
        assert!(
            names.contains(&"profi_cuda_calls_total"),
            "profi_cuda_calls_total present, got {:?}",
            names
        );
        assert!(
            names.contains(&"profi_cuda_duration_seconds"),
            "histogram present, got {:?}",
            names
        );
        assert!(
            !names.contains(&"profi_system_prometheus_encode_duration_seconds"),
            "profi_system_* self-obs metrics must be dropped by should_skip_metric, got {:?}",
            names
        );

        // service.name came from OtlpArgs.service_name, via build_resource.
        let service_name_attr = rm
            .resource
            .as_ref()
            .expect("resource")
            .attributes
            .iter()
            .find(|kv| kv.key == "service.name")
            .expect("service.name attr");
        if let Some(opentelemetry_proto::tonic::common::v1::any_value::Value::StringValue(v)) =
            service_name_attr
                .value
                .as_ref()
                .and_then(|v| v.value.clone())
        {
            assert_eq!(v, "profi-test");
        } else {
            panic!("service.name not a string: {:?}", service_name_attr);
        }

        // Label rename: `pod` → `k8s.pod.name` on the counter data point.
        let counter = metrics_proto
            .iter()
            .find(|m| m.name == "profi_cuda_calls_total")
            .unwrap();
        use opentelemetry_proto::tonic::metrics::v1::metric::Data;
        let sum = match counter.data.as_ref().expect("data") {
            Data::Sum(s) => s,
            other => panic!("expected Sum, got {:?}", other),
        };
        assert!(sum.is_monotonic);
        let dp = &sum.data_points[0];
        let keys: Vec<&str> = dp.attributes.iter().map(|kv| kv.key.as_str()).collect();
        assert!(
            keys.contains(&"k8s.pod.name"),
            "pod label renamed to k8s.pod.name, got {:?}",
            keys
        );

        // Histogram invariants: bucket_counts length == explicit_bounds length + 1,
        // and sum(bucket_counts) == count.
        let hist = metrics_proto
            .iter()
            .find(|m| m.name == "profi_cuda_duration_seconds")
            .unwrap();
        let hist_data = match hist.data.as_ref().expect("data") {
            Data::Histogram(h) => h,
            other => panic!("expected Histogram, got {:?}", other),
        };
        let hdp = &hist_data.data_points[0];
        assert_eq!(
            hdp.bucket_counts.len(),
            hdp.explicit_bounds.len() + 1,
            "N counts / N-1 bounds invariant"
        );
        assert_eq!(
            hdp.bucket_counts.iter().sum::<u64>(),
            hdp.count,
            "bucket_counts sum == count"
        );
    }

    let _ = shutdown_tx.send(());
    let _ = server_handle.await;
}
