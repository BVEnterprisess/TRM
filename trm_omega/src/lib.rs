pub mod deq;
pub mod network;
pub mod preset;
pub mod recursion;
pub mod augment;
pub mod data;
pub mod memory;
pub mod quantize;
pub mod kernel_ref;
pub mod kernel_dispatch;
pub mod trmq10;
pub mod task_eval;
pub mod train;
pub mod setup;

#[cfg(feature = "cuda")]
pub mod custom_kernels;

use candle_core::{Device, Result as CResult};

/// Select the best available device, with compute capability check.
pub fn default_device() -> CResult<Device> {
    #[cfg(feature = "candle-cuda")]
    {
        if let Ok(d) = Device::new_cuda(0) {
            return Ok(d);
        } else {
            log::warn!("CUDA unavailable, falling back to CPU");
        }
    }
    Ok(Device::Cpu)
}
