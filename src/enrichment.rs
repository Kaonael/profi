// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use arc_swap::ArcSwap;
use futures::TryStreamExt;
use kube::runtime::watcher::{self, Event};
use kube::runtime::WatchStreamExt;
use lasso::{Spur, ThreadedRodeo};

use log::{info, warn};

#[derive(Clone, Copy)]
pub struct Labels {
    pub comm: Spur,
    pub namespace: Spur,
    pub pod: Spur,
    pub container: Spur,
    pub gpu: Spur,
    pub gpu_uuid: Spur,
}

#[derive(Clone)]
pub struct GpuDevice {
    pub index: u32,
    pub uuid: String,
    pub name: String,
}

#[derive(Clone)]
struct PodInfo {
    namespace: String,
    pod: String,
    container: String,
    /// True when the pod carries annotation `profi/mode: full` —
    /// all its PIDs should be promoted to full kernel tracing regardless of
    /// the cluster-wide `--kernel-mode`. See `Enricher::upgraded_pids`.
    upgrade: bool,
}

/// K8s annotation that promotes a pod to full kernel tracing.
/// Value "full" upgrades; any other value (or missing) leaves the pod on the
/// cluster-wide `--kernel-mode` setting.
const MODE_ANNOTATION: &str = "profi/mode";

pub struct Enricher {
    proc_path: String,
    k8s_pid_cache: RwLock<HashMap<u32, Option<PodInfo>>>,
    gpu_pid_cache: RwLock<HashMap<u32, Option<GpuDevice>>>,
    container_map: ArcSwap<HashMap<String, PodInfo>>,
    pub gpu_devices: Vec<GpuDevice>,
    minor_to_gpu: HashMap<u32, GpuDevice>,
    pub changed_pids: Arc<std::sync::Mutex<HashSet<u32>>>,
    pub has_changes: AtomicBool,
    container_to_pids: RwLock<HashMap<String, HashSet<u32>>>,
    /// PIDs currently promoted to full kernel tracing via
    /// `profi/mode: full` pod annotation. Ground truth for the
    /// `UPGRADED_PIDS` eBPF map; main.rs drains this set periodically and
    /// replaces the map contents.
    pub upgraded_pids: RwLock<HashSet<u32>>,
    pub upgrade_dirty: AtomicBool,
    pub interner: ThreadedRodeo,
}

impl Enricher {
    pub fn new(proc_path: String) -> Arc<Self> {
        let gpu_devices = discover_gpu_devices(&proc_path);
        if gpu_devices.is_empty() {
            warn!(
                "no NVIDIA GPUs found in {proc_path}/driver/nvidia/gpus — GPU enrichment disabled"
            );
        } else {
            info!("procfs: {} GPU(s) found", gpu_devices.len());
        }
        let minor_to_gpu: HashMap<u32, GpuDevice> =
            gpu_devices.iter().map(|d| (d.index, d.clone())).collect();
        Arc::new(Self {
            proc_path,
            k8s_pid_cache: RwLock::new(HashMap::new()),
            gpu_pid_cache: RwLock::new(HashMap::new()),
            container_map: ArcSwap::from_pointee(HashMap::new()),
            gpu_devices,
            minor_to_gpu,
            changed_pids: Arc::new(std::sync::Mutex::new(HashSet::new())),
            has_changes: AtomicBool::new(false),
            container_to_pids: RwLock::new(HashMap::new()),
            upgraded_pids: RwLock::new(HashSet::new()),
            upgrade_dirty: AtomicBool::new(false),
            interner: ThreadedRodeo::default(),
        })
    }

    pub fn lookup(&self, pid: u32, comm: &str) -> Labels {
        let k8s = self.resolve_k8s(pid);
        let gpu = self.resolve_gpu(pid);

        let empty = self.interner.get_or_intern_static("");

        Labels {
            comm: self.interner.get_or_intern(comm),
            namespace: k8s
                .as_ref()
                .map_or(empty, |k| self.interner.get_or_intern(&k.namespace)),
            pod: k8s
                .as_ref()
                .map_or(empty, |k| self.interner.get_or_intern(&k.pod)),
            container: k8s
                .as_ref()
                .map_or(empty, |k| self.interner.get_or_intern(&k.container)),
            gpu: gpu
                .as_ref()
                .map_or(empty, |g| self.interner.get_or_intern(g.index.to_string())),
            gpu_uuid: gpu
                .as_ref()
                .map_or(empty, |g| self.interner.get_or_intern(&g.uuid)),
        }
    }

    pub fn evict_pid(&self, pid: u32) {
        self.k8s_pid_cache.write().unwrap().remove(&pid);
        self.gpu_pid_cache.write().unwrap().remove(&pid);
        if self.upgraded_pids.write().unwrap().remove(&pid) {
            self.upgrade_dirty.store(true, Ordering::Release);
        }
    }

    fn resolve_k8s(&self, pid: u32) -> Option<PodInfo> {
        if let Some(cached) = self.k8s_pid_cache.read().unwrap().get(&pid) {
            return cached.clone();
        }

        let cgroup_path = format!("{}/{}/cgroup", self.proc_path, pid);
        let cgroup = std::fs::read_to_string(cgroup_path).ok()?;
        let container_id = extract_container_id(&cgroup);

        let result = container_id.as_ref().and_then(|id| {
            let map = self.container_map.load();
            map.get(id).cloned()
        });

        if let Some(cid) = container_id {
            self.container_to_pids
                .write()
                .unwrap()
                .entry(cid)
                .or_default()
                .insert(pid);
        }

        // Annotation-driven upgrade: if this PID belongs to a pod with
        // `profi/mode: full`, record it so the eBPF UPGRADED_PIDS
        // map picks it up on the next sync tick.
        if let Some(info) = &result {
            if info.upgrade && self.upgraded_pids.write().unwrap().insert(pid) {
                self.upgrade_dirty.store(true, Ordering::Release);
            }
        }

        self.k8s_pid_cache
            .write()
            .unwrap()
            .insert(pid, result.clone());
        result
    }

    fn invalidate_containers(&self, container_ids: &[String]) {
        if container_ids.is_empty() {
            return;
        }
        let c2p = self.container_to_pids.read().unwrap();
        let mut affected_pids = HashSet::new();
        for cid in container_ids {
            if let Some(pids) = c2p.get(cid) {
                affected_pids.extend(pids.iter().copied());
            }
        }
        drop(c2p);

        if !affected_pids.is_empty() {
            let mut pid_cache = self.k8s_pid_cache.write().unwrap();
            for pid in &affected_pids {
                pid_cache.remove(pid);
            }
            drop(pid_cache);

            // Drop annotation-upgrade flag for these PIDs; the next
            // resolve_k8s() call will re-add them if the pod still carries
            // the annotation. Ensures deleted / re-annotated pods don't
            // leave stale entries in the eBPF UPGRADED_PIDS map.
            {
                let mut upg = self.upgraded_pids.write().unwrap();
                let before = upg.len();
                for pid in &affected_pids {
                    upg.remove(pid);
                }
                if upg.len() != before {
                    self.upgrade_dirty.store(true, Ordering::Release);
                }
            }

            let mut changed = self.changed_pids.lock().unwrap();
            changed.extend(affected_pids);
            self.has_changes.store(true, Ordering::Release);
        }
    }

    pub fn start_k8s_refresh(self: &Arc<Self>, node_name: String, _interval: std::time::Duration) {
        let enricher = self.clone();
        tokio::spawn(async move {
            let client = match kube::Client::try_default().await {
                Ok(c) => {
                    info!("K8s client connected — pod enrichment enabled");
                    c
                }
                Err(e) => {
                    warn!("K8s client unavailable: {e} — pod enrichment disabled");
                    return;
                }
            };

            let pods_api: kube::Api<k8s_openapi::api::core::v1::Pod> = kube::Api::all(client);

            let config = watcher::Config::default().fields(&format!("spec.nodeName={node_name}"));

            let mut stream = Box::pin(watcher::watcher(pods_api, config).default_backoff());

            let mut pending_map: HashMap<String, PodInfo> = HashMap::new();

            loop {
                match stream.try_next().await {
                    Ok(Some(event)) => match event {
                        Event::Init => {
                            pending_map.clear();
                        }
                        Event::InitApply(pod) => {
                            let containers = extract_containers_from_pod(&pod);
                            pending_map.extend(containers);
                        }
                        Event::InitDone => {
                            let old_map =
                                enricher.container_map.swap(Arc::new(pending_map.clone()));
                            let removed: Vec<String> = old_map
                                .keys()
                                .filter(|k| !pending_map.contains_key(*k))
                                .cloned()
                                .collect();
                            enricher.invalidate_containers(&removed);

                            {
                                let mut pid_cache = enricher.k8s_pid_cache.write().unwrap();
                                pid_cache.retain(|_, v| v.is_some());
                            }

                            info!(
                                "K8s watch init done: {} containers mapped",
                                pending_map.len()
                            );
                            pending_map.clear();
                        }
                        Event::Apply(pod) => {
                            let new_containers = extract_containers_from_pod(&pod);
                            let pod_name = pod.metadata.name.unwrap_or_default();
                            let pod_ns = pod.metadata.namespace.unwrap_or_default();

                            let old_cids: Vec<String> = {
                                let map = enricher.container_map.load();
                                map.iter()
                                    .filter(|(_, info)| {
                                        info.pod == pod_name && info.namespace == pod_ns
                                    })
                                    .map(|(cid, _)| cid.clone())
                                    .collect()
                            };

                            let old_cids_clone = old_cids.clone();
                            enricher.container_map.rcu(move |old| {
                                let mut new = (**old).clone();
                                for cid in &old_cids_clone {
                                    new.remove(cid);
                                }
                                new.extend(new_containers.clone());
                                new
                            });

                            enricher.invalidate_containers(&old_cids);
                        }
                        Event::Delete(pod) => {
                            let pod_name = pod.metadata.name.unwrap_or_default();
                            let pod_ns = pod.metadata.namespace.unwrap_or_default();

                            let removed_cids: Vec<String> = {
                                let map = enricher.container_map.load();
                                map.iter()
                                    .filter(|(_, info)| {
                                        info.pod == pod_name && info.namespace == pod_ns
                                    })
                                    .map(|(cid, _)| cid.clone())
                                    .collect()
                            };

                            if !removed_cids.is_empty() {
                                let removed_clone = removed_cids.clone();
                                enricher.container_map.rcu(move |old| {
                                    let mut new = (**old).clone();
                                    for cid in &removed_clone {
                                        new.remove(cid);
                                    }
                                    new
                                });
                            }

                            enricher.invalidate_containers(&removed_cids);
                        }
                    },
                    Ok(None) => {
                        warn!("K8s watch stream ended unexpectedly");
                        break;
                    }
                    Err(e) => {
                        warn!("K8s watch error: {e}");
                    }
                }
            }
        });
    }

    fn resolve_gpu(&self, pid: u32) -> Option<GpuDevice> {
        if let Some(cached) = self.gpu_pid_cache.read().unwrap().get(&pid) {
            return cached.clone();
        }

        if self.gpu_devices.is_empty() {
            self.gpu_pid_cache.write().unwrap().insert(pid, None);
            return None;
        }

        let result = resolve_pid_gpu(&self.proc_path, pid, &self.minor_to_gpu);
        self.gpu_pid_cache
            .write()
            .unwrap()
            .insert(pid, result.clone());
        result
    }
}

fn extract_containers_from_pod(pod: &k8s_openapi::api::core::v1::Pod) -> HashMap<String, PodInfo> {
    let mut map = HashMap::new();
    let ns = pod.metadata.namespace.clone().unwrap_or_default();
    let name = pod.metadata.name.clone().unwrap_or_default();
    let upgrade = pod
        .metadata
        .annotations
        .as_ref()
        .and_then(|a| a.get(MODE_ANNOTATION))
        .is_some_and(|v| v == "full");
    if let Some(status) = &pod.status {
        let all_statuses = status
            .container_statuses
            .iter()
            .flatten()
            .chain(status.init_container_statuses.iter().flatten());
        for cs in all_statuses {
            if let Some(id) = &cs.container_id {
                if let Some(short) = short_container_id(id) {
                    map.insert(
                        short,
                        PodInfo {
                            namespace: ns.clone(),
                            pod: name.clone(),
                            container: cs.name.clone(),
                            upgrade,
                        },
                    );
                }
            }
        }
    }
    map
}

fn discover_gpu_devices(proc_path: &str) -> Vec<GpuDevice> {
    let gpu_dir = format!("{proc_path}/driver/nvidia/gpus");
    let entries = match std::fs::read_dir(&gpu_dir) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };

    let mut devices = Vec::new();
    for entry in entries.flatten() {
        let info_path = entry.path().join("information");
        let info = match std::fs::read_to_string(&info_path) {
            Ok(s) => s,
            Err(_) => continue,
        };

        let mut index = None;
        let mut uuid = String::new();
        let mut name = String::new();

        for line in info.lines() {
            if let Some(val) = line.strip_prefix("Device Minor:") {
                index = val.trim().parse().ok();
            } else if let Some(val) = line.strip_prefix("GPU UUID:") {
                uuid = val.trim().to_string();
            } else if let Some(val) = line.strip_prefix("Model:") {
                name = val.trim().to_string();
            }
        }

        if let Some(idx) = index {
            devices.push(GpuDevice {
                index: idx,
                uuid,
                name,
            });
        }
    }

    devices.sort_by_key(|d| d.index);
    devices
}

fn resolve_pid_gpu(
    proc_path: &str,
    pid: u32,
    minor_to_gpu: &HashMap<u32, GpuDevice>,
) -> Option<GpuDevice> {
    let fd_dir = format!("{proc_path}/{pid}/fd");
    let fds = match std::fs::read_dir(&fd_dir) {
        Ok(e) => e,
        Err(_) => return None,
    };

    let mut dev_counts: HashMap<u32, u32> = HashMap::new();
    for fd in fds.flatten() {
        let link = match std::fs::read_link(fd.path()) {
            Ok(l) => l,
            Err(_) => continue,
        };
        let link_str = link.to_string_lossy();
        if let Some(rest) = link_str.strip_prefix("/dev/nvidia") {
            if let Ok(minor) = rest.parse::<u32>() {
                if minor_to_gpu.contains_key(&minor) {
                    *dev_counts.entry(minor).or_insert(0) += 1;
                }
            }
        }
    }

    dev_counts
        .iter()
        .max_by(|a, b| a.1.cmp(b.1).then(a.0.cmp(b.0)))
        .and_then(|(&minor, _)| minor_to_gpu.get(&minor).cloned())
}

fn extract_container_id(cgroup: &str) -> Option<String> {
    for line in cgroup.lines() {
        if let Some(idx) = line.find("cri-containerd-") {
            let rest = &line[idx + 15..];
            if let Some(id) = rest.split('.').next() {
                if id.len() >= 12 {
                    return Some(id[..12].to_string());
                }
            }
        }
        if let Some(idx) = line.find("crio-") {
            let rest = &line[idx + 5..];
            if let Some(id) = rest.split('.').next() {
                if id.len() >= 12 {
                    return Some(id[..12].to_string());
                }
            }
        }
        // Match last path segment if it looks like a 64-hex container ID.
        // Covers both "/kubepods/.../ID" and relative "/../../../.../ID" forms.
        if let Some(last_slash) = line.rfind('/') {
            let segment = &line[last_slash + 1..];
            let id = segment.split('.').next().unwrap_or(segment);
            if id.len() >= 64 && id.chars().all(|c| c.is_ascii_hexdigit()) {
                return Some(id[..12].to_string());
            }
        }
    }
    None
}

fn short_container_id(full_id: &str) -> Option<String> {
    let raw = full_id.rsplit("://").next()?;
    if raw.len() >= 12 {
        Some(raw[..12].to_string())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_containerd() {
        let cgroup = "0::/kubepods/besteffort/pod123/cri-containerd-abc123def45678.scope";
        assert_eq!(
            extract_container_id(cgroup),
            Some("abc123def456".to_string())
        );
    }

    #[test]
    fn extract_crio() {
        let cgroup = "0::/kubepods/besteffort/pod123/crio-abc123def45678.scope";
        assert_eq!(
            extract_container_id(cgroup),
            Some("abc123def456".to_string())
        );
    }

    #[test]
    fn extract_hex_64_generic() {
        let id = "a".repeat(64);
        let cgroup = format!("12:memory:/kubepods/besteffort/pod123/{id}");
        assert_eq!(
            extract_container_id(&cgroup),
            Some("aaaaaaaaaaaa".to_string())
        );
    }

    #[test]
    fn extract_no_container() {
        assert_eq!(extract_container_id("0::/"), None);
    }

    #[test]
    fn extract_multiline_second_match() {
        let cgroup = "0::/\n0::/kubepods/pod/cri-containerd-deadbeefcafe1234.scope";
        assert_eq!(
            extract_container_id(cgroup),
            Some("deadbeefcafe".to_string())
        );
    }

    #[test]
    fn extract_short_id_none() {
        let cgroup = "0::/kubepods/pod/cri-containerd-abc.scope";
        assert_eq!(extract_container_id(cgroup), None);
    }

    #[test]
    fn extract_non_hex_no_match() {
        let id = format!("{}zzzz", "a".repeat(60));
        let cgroup = format!("0::/kubepods/{id}");
        assert_eq!(extract_container_id(&cgroup), None);
    }

    #[test]
    fn extract_docker_cgroup_v1() {
        let id = "a".repeat(64);
        let cgroup = format!("12:memory:/docker/{id}");
        assert_eq!(
            extract_container_id(&cgroup),
            Some("aaaaaaaaaaaa".to_string())
        );
    }

    #[test]
    fn short_containerd_protocol() {
        assert_eq!(
            short_container_id("containerd://abc123def456789xyz"),
            Some("abc123def456".to_string())
        );
    }

    #[test]
    fn short_too_short() {
        assert_eq!(short_container_id("short"), None);
    }

    #[test]
    fn short_docker_protocol() {
        assert_eq!(
            short_container_id("docker://abcdef012345"),
            Some("abcdef012345".to_string())
        );
    }

    fn pod_with_annotation(
        value: Option<&str>,
        container_id: &str,
    ) -> k8s_openapi::api::core::v1::Pod {
        use k8s_openapi::api::core::v1::{ContainerStatus, PodStatus};
        use kube::core::ObjectMeta;
        use std::collections::BTreeMap;
        let mut annotations = BTreeMap::new();
        if let Some(v) = value {
            annotations.insert(MODE_ANNOTATION.to_string(), v.to_string());
        }
        k8s_openapi::api::core::v1::Pod {
            metadata: ObjectMeta {
                name: Some("demo".to_string()),
                namespace: Some("ns".to_string()),
                annotations: if annotations.is_empty() {
                    None
                } else {
                    Some(annotations)
                },
                ..Default::default()
            },
            status: Some(PodStatus {
                container_statuses: Some(vec![ContainerStatus {
                    name: "demo".to_string(),
                    container_id: Some(format!("containerd://{}", container_id)),
                    ..Default::default()
                }]),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn annotation_full_marks_upgrade() {
        let id = "a".repeat(64);
        let pod = pod_with_annotation(Some("full"), &id);
        let info = extract_containers_from_pod(&pod)
            .into_values()
            .next()
            .expect("one container expected");
        assert!(info.upgrade);
    }

    #[test]
    fn annotation_missing_leaves_upgrade_false() {
        let id = "b".repeat(64);
        let pod = pod_with_annotation(None, &id);
        let info = extract_containers_from_pod(&pod)
            .into_values()
            .next()
            .unwrap();
        assert!(!info.upgrade);
    }

    #[test]
    fn annotation_unknown_value_does_not_upgrade() {
        // Only the literal "full" upgrades; other values (including "anonymous",
        // "yes", "true", etc.) are ignored to avoid typo-driven overhead.
        let id = "c".repeat(64);
        let pod = pod_with_annotation(Some("anonymous"), &id);
        let info = extract_containers_from_pod(&pod)
            .into_values()
            .next()
            .unwrap();
        assert!(!info.upgrade);

        let pod2 = pod_with_annotation(Some("true"), &id);
        let info2 = extract_containers_from_pod(&pod2)
            .into_values()
            .next()
            .unwrap();
        assert!(!info2.upgrade);
    }
}
