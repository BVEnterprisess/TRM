//! Named architectures. Training scales dim; depth stays 2.
//!
//! | preset | dim | heads | layers | expected live params |
//! |--------|-----|-------|--------|----------------------|
//! | tiny   | 32  | 4     | 2      | ~0.16M (smoke) |
//! | paper  | 256 | 8     | 2      | 2.67M (measured) |
//! | 7m     | 448 | 8     | 2      | 6,787,969 (6.788M, instantiated) |

use anyhow::{anyhow, Result};

use crate::network::NetworkConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Architecture {
    pub name: &'static str,
    pub dim: usize,
    pub heads: usize,
    pub layers: usize,
    pub vocab: usize,
}

impl Architecture {
    pub fn apply_network(&self, cfg: &mut NetworkConfig) {
        cfg.dim = self.dim;
        cfg.num_heads = self.heads;
        cfg.num_layers = self.layers;
        cfg.vocab_size = self.vocab;
    }
}

pub fn tiny() -> Architecture {
    Architecture {
        name: "tiny",
        dim: 32,
        heads: 4,
        layers: 2,
        vocab: 11,
    }
}

pub fn paper() -> Architecture {
    Architecture {
        name: "paper",
        dim: 256,
        heads: 8,
        layers: 2,
        vocab: 11,
    }
}

/// Closest CLI-legal ~7M: dim 448, 8 heads, 2 layers, 4096 puzzle embeddings.
/// Depth stays 2. Scaling is width, not depth.
pub fn seven_m() -> Architecture {
    Architecture {
        name: "7m",
        dim: 448,
        heads: 8,
        layers: 2,
        vocab: 11,
    }
}

pub fn parse(name: &str) -> Result<Architecture> {
    match name.trim().to_ascii_lowercase().as_str() {
        "tiny" | "smoke" => Ok(tiny()),
        "paper" | "default" | "2.67m" => Ok(paper()),
        "7m" | "7" | "seven" => Ok(seven_m()),
        other => Err(anyhow!(
            "unknown preset '{other}'; expected tiny, paper, or 7m"
        )),
    }
}

/// `--preset` overrides dim/heads/layers. Without a preset, CLI values win.
pub fn resolve(
    preset: Option<&str>,
    dim: usize,
    heads: usize,
    layers: usize,
) -> Result<(usize, usize, usize)> {
    match preset {
        None => Ok((dim, heads, layers)),
        Some(name) => {
            let a = parse(name)?;
            Ok((a.dim, a.heads, a.layers))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_without_preset_keeps_cli() {
        assert_eq!(resolve(None, 99, 3, 2).unwrap(), (99, 3, 2));
    }

    #[test]
    fn resolve_preset_overrides_cli_dim() {
        assert_eq!(resolve(Some("7m"), 256, 8, 2).unwrap(), (448, 8, 2));
    }
}
