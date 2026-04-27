// SPDX-License-Identifier: Apache-2.0

use log::{info, warn};
use nvml_wrapper::bitmasks::device::ThrottleReasons;
use nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor};
use nvml_wrapper::Nvml;

use crate::metrics::Metrics;

struct DeviceInfo {
    index: u32,
    uuid: String,
    index_str: String,
}

pub struct NvmlPoller {
    nvml: Nvml,
    devices: Vec<DeviceInfo>,
    /// Previous ECC error counts for delta computation.
    prev_ecc: Vec<[u64; 4]>, // [corrected_sram, corrected_dram, uncorrected_sram, uncorrected_dram]
}

impl NvmlPoller {
    /// Try to initialize NVML. Returns None if libnvidia-ml.so is not available.
    pub fn new() -> Option<Self> {
        let nvml = match Nvml::init() {
            Ok(n) => n,
            Err(e) => {
                info!("NVML init failed ({e}) — GPU hardware metrics disabled");
                return None;
            }
        };

        let count = match nvml.device_count() {
            Ok(c) => c,
            Err(e) => {
                warn!("NVML device_count failed: {e}");
                return None;
            }
        };

        let mut devices = Vec::with_capacity(count as usize);
        for i in 0..count {
            match nvml.device_by_index(i) {
                Ok(dev) => {
                    let uuid = dev.uuid().unwrap_or_else(|_| format!("unknown-{i}"));
                    devices.push(DeviceInfo {
                        index: i,
                        uuid,
                        index_str: i.to_string(),
                    });
                }
                Err(e) => warn!("NVML: cannot access GPU {i}: {e}"),
            }
        }

        info!("NVML initialized: {} GPU(s)", devices.len());
        let prev_ecc = vec![[0u64; 4]; devices.len()];
        Some(Self {
            nvml,
            devices,
            prev_ecc,
        })
    }

    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Poll all GPU devices and update Prometheus metrics.
    pub fn poll(&mut self, metrics: &Metrics) {
        for dev_idx in 0..self.devices.len() {
            let gpu_index = self.devices[dev_idx].index;
            let gpu = self.devices[dev_idx].index_str.clone();
            let uuid = self.devices[dev_idx].uuid.clone();
            let dev = match self.nvml.device_by_index(gpu_index) {
                Ok(d) => d,
                Err(_) => continue,
            };
            let gpu = gpu.as_str();
            let uuid = uuid.as_str();

            // Temperature
            if let Ok(temp) = dev.temperature(TemperatureSensor::Gpu) {
                metrics
                    .gpu_temperature
                    .with_label_values(&[gpu, uuid])
                    .set(temp as f64);
            }

            // Power (milliwatts → watts)
            if let Ok(power_mw) = dev.power_usage() {
                metrics
                    .gpu_power
                    .with_label_values(&[gpu, uuid])
                    .set(power_mw as f64 / 1000.0);
            }

            // Clocks
            if let Ok(sm) = dev.clock_info(Clock::SM) {
                metrics
                    .gpu_clock
                    .with_label_values(&[gpu, uuid, "sm"])
                    .set(sm as f64);
            }
            if let Ok(mem) = dev.clock_info(Clock::Memory) {
                metrics
                    .gpu_clock
                    .with_label_values(&[gpu, uuid, "mem"])
                    .set(mem as f64);
            }

            // Utilization
            if let Ok(util) = dev.utilization_rates() {
                metrics
                    .gpu_utilization
                    .with_label_values(&[gpu, uuid, "gpu"])
                    .set(util.gpu as f64 / 100.0);
                metrics
                    .gpu_utilization
                    .with_label_values(&[gpu, uuid, "memory"])
                    .set(util.memory as f64 / 100.0);
            }

            // Memory
            if let Ok(mem) = dev.memory_info() {
                metrics
                    .gpu_memory
                    .with_label_values(&[gpu, uuid, "used"])
                    .set(mem.used as f64);
                metrics
                    .gpu_memory
                    .with_label_values(&[gpu, uuid, "free"])
                    .set(mem.free as f64);
                metrics
                    .gpu_memory
                    .with_label_values(&[gpu, uuid, "total"])
                    .set(mem.total as f64);
            }

            // ECC errors (volatile = since last driver load)
            {
                use nvml_wrapper::enum_wrappers::device::{EccCounter, MemoryError};
                let types: &[(MemoryError, &str, usize)] = &[
                    (MemoryError::Corrected, "corrected", 0),
                    (MemoryError::Uncorrected, "uncorrected", 1),
                ];
                for &(err_type, type_label, ecc_idx) in types {
                    if let Ok(count) = dev.total_ecc_errors(err_type, EccCounter::Volatile) {
                        let prev = self.prev_ecc[dev_idx][ecc_idx];
                        if count > prev {
                            metrics
                                .gpu_ecc_errors
                                .with_label_values(&[gpu, uuid, type_label, "all"])
                                .inc_by(count - prev);
                        }
                        self.prev_ecc[dev_idx][ecc_idx] = count;
                    }
                }
            }

            // Throttle reasons
            if let Ok(reasons) = dev.current_throttle_reasons() {
                let checks: &[(&str, ThrottleReasons)] = &[
                    ("idle", ThrottleReasons::GPU_IDLE),
                    ("power", ThrottleReasons::SW_POWER_CAP),
                    ("thermal", ThrottleReasons::SW_THERMAL_SLOWDOWN),
                    ("hw_thermal", ThrottleReasons::HW_THERMAL_SLOWDOWN),
                    ("hw_power", ThrottleReasons::HW_POWER_BRAKE_SLOWDOWN),
                    ("sync_boost", ThrottleReasons::SYNC_BOOST),
                ];
                for &(label, flag) in checks {
                    let active = if reasons.contains(flag) { 1.0 } else { 0.0 };
                    metrics
                        .gpu_throttle
                        .with_label_values(&[gpu, uuid, label])
                        .set(active);
                }
            }
        }
    }
}
