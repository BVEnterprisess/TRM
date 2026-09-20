//! TRMQ10 deployment bundle: 2-bit ternary 2D weights + FP32 leftovers.
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use anyhow::{bail, Context, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use candle_core::{DType, Device, Tensor};

use crate::network::NetworkConfig;
use crate::quantize::{quantize_rowwise_ternary, unpack_ternary, TernaryPacked};
use crate::recursion::{TrmConfig, TrmModel};

pub const MAGIC: &[u8; 6] = b"TRMQ10";
pub const VERSION: u16 = 1;

#[derive(Debug, Clone)]
pub struct TrmqHeader {
    pub dim: u32,
    pub heads: u32,
    pub max_seq: u32,
    pub vocab: u32,
    pub layers: u32,
    pub z_seq: u32,
    pub n_l: u32,
    pub n_sup: u32,
    pub max_puzzles: u32,
    pub flags: u32,
}

impl TrmqHeader {
    pub fn from_model(model: &TrmModel) -> Self {
        let mut flags = 0u32;
        if model.trm_cfg.use_learned_halt_head {
            flags |= 1;
        }
        Self {
            dim: model.net_cfg.dim as u32,
            heads: model.net_cfg.num_heads as u32,
            max_seq: model.net_cfg.max_seq_len as u32,
            vocab: model.net_cfg.vocab_size as u32,
            layers: model.net_cfg.num_layers as u32,
            z_seq: model.trm_cfg.z_seq as u32,
            n_l: model.trm_cfg.n_l_cycles as u32,
            n_sup: model.trm_cfg.n_sup as u32,
            max_puzzles: model.trm_cfg.max_puzzles as u32,
            flags,
        }
    }

    pub fn network_config(&self) -> NetworkConfig {
        NetworkConfig {
            dim: self.dim as usize,
            num_heads: self.heads as usize,
            max_seq_len: self.max_seq as usize,
            vocab_size: self.vocab as usize,
            num_layers: self.layers as usize,
            ..Default::default()
        }
    }

    pub fn trm_config(&self, use_deq: bool) -> TrmConfig {
        TrmConfig {
            z_seq: self.z_seq as usize,
            n_l_cycles: self.n_l as usize,
            n_sup: self.n_sup as usize,
            max_puzzles: self.max_puzzles as usize,
            use_learned_halt_head: self.flags & 1 != 0,
            use_deq,
            ..Default::default()
        }
    }
}

fn can_pack(t: &Tensor) -> bool {
    t.rank() == 2 && t.dims()[1] % 4 == 0 && t.dims()[0] > 0
}

/// Write a self-describing TRMQ10 bundle from a live model.
pub fn save_trmq10<P: AsRef<Path>>(path: P, model: &TrmModel) -> Result<()> {
    let named = model.load_named_tensors();
    let header = TrmqHeader::from_model(model);

    let mut quantized = BTreeMap::new();
    let mut fp32 = BTreeMap::new();
    for (name, t) in &named {
        if can_pack(t) {
            let (out_dim, in_dim) = t.dims2().context(name.clone())?;
            let data = t.flatten_all()?.to_vec1::<f32>()?;
            quantized.insert(
                name.clone(),
                quantize_rowwise_ternary(&data, out_dim, in_dim),
            );
        } else {
            fp32.insert(name.clone(), t.clone());
        }
    }

    let mut out = BufWriter::new(File::create(path.as_ref())?);
    out.write_all(MAGIC)?;
    out.write_u16::<LittleEndian>(VERSION)?;
    out.write_u32::<LittleEndian>(header.dim)?;
    out.write_u32::<LittleEndian>(header.heads)?;
    out.write_u32::<LittleEndian>(header.max_seq)?;
    out.write_u32::<LittleEndian>(header.vocab)?;
    out.write_u32::<LittleEndian>(header.layers)?;
    out.write_u32::<LittleEndian>(header.z_seq)?;
    out.write_u32::<LittleEndian>(header.n_l)?;
    out.write_u32::<LittleEndian>(header.n_sup)?;
    out.write_u32::<LittleEndian>(header.max_puzzles)?;
    out.write_u32::<LittleEndian>(header.flags)?;

    out.write_u32::<LittleEndian>(quantized.len() as u32)?;
    for (name, pack) in &quantized {
        write_name(&mut out, name)?;
        out.write_u32::<LittleEndian>(pack.out_dim as u32)?;
        out.write_u32::<LittleEndian>(pack.in_dim as u32)?;
        out.write_u32::<LittleEndian>(pack.alphas.len() as u32)?;
        out.write_all(bytemuck::cast_slice(&pack.alphas))?;
        out.write_u32::<LittleEndian>(pack.packed.len() as u32)?;
        out.write_all(&pack.packed)?;
    }

    out.write_u32::<LittleEndian>(fp32.len() as u32)?;
    for (name, t) in &fp32 {
        write_name(&mut out, name)?;
        let dims = t.dims();
        out.write_u32::<LittleEndian>(dims.len() as u32)?;
        for d in dims {
            out.write_u32::<LittleEndian>(*d as u32)?;
        }
        let data = t.flatten_all()?.to_vec1::<f32>()?;
        out.write_u32::<LittleEndian>(data.len() as u32)?;
        out.write_all(bytemuck::cast_slice(&data))?;
    }
    out.flush()?;
    log::info!(
        "forged {} ({} ternary, {} fp32)",
        path.as_ref().display(),
        quantized.len(),
        fp32.len()
    );
    Ok(())
}

pub fn is_trmq10<P: AsRef<Path>>(path: P) -> bool {
    let Ok(mut f) = File::open(path.as_ref()) else {
        return false;
    };
    let mut mag = [0u8; 6];
    f.read_exact(&mut mag).is_ok() && &mag == MAGIC
}

/// Load safetensors or TRMQ10 based on magic/extension.
pub fn load_model<P: AsRef<Path>>(
    device: Device,
    path: P,
    fallback_net: NetworkConfig,
    fallback_trm: TrmConfig,
) -> Result<TrmModel> {
    let path = path.as_ref();
    if is_trmq10(path) {
        load_trmq10(device, path, fallback_trm.use_deq)
    } else {
        TrmModel::load_safetensors(device, fallback_net, fallback_trm, path)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
}

pub fn load_trmq10<P: AsRef<Path>>(device: Device, path: P, use_deq: bool) -> Result<TrmModel> {
    let mut r = BufReader::new(File::open(path.as_ref())?);
    let mut mag = [0u8; 6];
    r.read_exact(&mut mag)?;
    if &mag != MAGIC {
        bail!("not a TRMQ10 file: {}", path.as_ref().display());
    }
    let version = r.read_u16::<LittleEndian>()?;
    if version != VERSION {
        bail!("unsupported TRMQ10 version {version}");
    }
    let header = TrmqHeader {
        dim: r.read_u32::<LittleEndian>()?,
        heads: r.read_u32::<LittleEndian>()?,
        max_seq: r.read_u32::<LittleEndian>()?,
        vocab: r.read_u32::<LittleEndian>()?,
        layers: r.read_u32::<LittleEndian>()?,
        z_seq: r.read_u32::<LittleEndian>()?,
        n_l: r.read_u32::<LittleEndian>()?,
        n_sup: r.read_u32::<LittleEndian>()?,
        max_puzzles: r.read_u32::<LittleEndian>()?,
        flags: r.read_u32::<LittleEndian>()?,
    };

    let n_q = r.read_u32::<LittleEndian>()? as usize;
    let mut quantized = BTreeMap::new();
    for _ in 0..n_q {
        let name = read_name(&mut r)?;
        let out_dim = r.read_u32::<LittleEndian>()? as usize;
        let in_dim = r.read_u32::<LittleEndian>()? as usize;
        let n_alpha = r.read_u32::<LittleEndian>()? as usize;
        let alphas = read_f32_vec(&mut r, n_alpha)?;
        let n_pack = r.read_u32::<LittleEndian>()? as usize;
        let mut packed = vec![0u8; n_pack];
        r.read_exact(&mut packed)?;
        quantized.insert(
            name,
            TernaryPacked {
                out_dim,
                in_dim,
                packed,
                alphas,
            },
        );
    }

    let n_fp = r.read_u32::<LittleEndian>()? as usize;
    let mut fp32 = BTreeMap::new();
    for _ in 0..n_fp {
        let name = read_name(&mut r)?;
        let ndim = r.read_u32::<LittleEndian>()? as usize;
        let mut dims = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            dims.push(r.read_u32::<LittleEndian>()? as usize);
        }
        let n = r.read_u32::<LittleEndian>()? as usize;
        let data = read_f32_vec(&mut r, n)?;
        let t = Tensor::from_vec(data, dims.as_slice(), &device)
            .with_context(|| format!("fp32 tensor {name}"))?;
        fp32.insert(name, t);
    }

    let net_cfg = header.network_config();
    let trm_cfg = header.trm_config(use_deq);
    let mut model = TrmModel::new(device.clone(), net_cfg, trm_cfg)
        .map_err(|e| anyhow::anyhow!("init model: {e}"))?;

    apply_named(&model, &quantized, &fp32, &device)?;
    model.apply_ternary_packs(&quantized);
    Ok(model)
}

fn apply_named(
    model: &TrmModel,
    quantized: &BTreeMap<String, TernaryPacked>,
    fp32: &BTreeMap<String, Tensor>,
    device: &Device,
) -> Result<()> {
    let data = model.varmap.data().lock().unwrap();
    for (name, pack) in quantized {
        let unpacked = unpack_ternary(pack);
        let t = Tensor::from_vec(unpacked, (pack.out_dim, pack.in_dim), device)
            .with_context(|| format!("unpack {name}"))?;
        let var = data
            .get(name)
            .with_context(|| format!("TRMQ10 tensor {name} not in model"))?;
        var.set(&t).map_err(|e| anyhow::anyhow!("set {name}: {e}"))?;
    }
    for (name, t) in fp32 {
        let var = data
            .get(name)
            .with_context(|| format!("TRMQ10 fp32 {name} not in model"))?;
        let t = t.to_device(device)?;
        let t = if t.dtype() != DType::F32 {
            t.to_dtype(DType::F32)?
        } else {
            t
        };
        var.set(&t)
            .map_err(|e| anyhow::anyhow!("set fp32 {name}: {e}"))?;
    }
    Ok(())
}

fn read_f32_vec<R: Read>(r: &mut R, n: usize) -> Result<Vec<f32>> {
    let mut bytes = vec![0u8; n * 4];
    r.read_exact(&mut bytes)?;
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn write_name<W: Write>(w: &mut W, name: &str) -> Result<()> {
    let b = name.as_bytes();
    w.write_u32::<LittleEndian>(b.len() as u32)?;
    w.write_all(b)?;
    Ok(())
}

fn read_name<R: Read>(r: &mut R) -> Result<String> {
    let n = r.read_u32::<LittleEndian>()? as usize;
    let mut b = vec![0u8; n];
    r.read_exact(&mut b)?;
    Ok(String::from_utf8(b)?)
}
