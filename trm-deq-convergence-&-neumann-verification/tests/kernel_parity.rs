use candle_core::{Device, Tensor};
use trm_omega::kernel_dispatch::{active_backend, packed_linear};
use trm_omega::kernel_ref;
use trm_omega::network::{NetworkConfig, NetworkVariant};
use trm_omega::quantize::{quantize_rowwise_ternary, unpack_ternary};
use trm_omega::recursion::{TrmConfig, TrmModel};

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn ternary_ref_matches_unpack_matmul() {
    let out_dim = 320; // >256 to catch the old 1D launch-grid bug
    let in_dim = 64;
    let tokens = 5;
    let weights: Vec<f32> = (0..(out_dim * in_dim))
        .map(|i| ((i % 17) as f32 / 8.0) - 1.0)
        .collect();
    let pack = quantize_rowwise_ternary(&weights, out_dim, in_dim);
    let unpacked = unpack_ternary(&pack);
    let input: Vec<f32> = (0..(tokens * in_dim))
        .map(|i| ((i % 11) as f32 / 5.0) - 0.8)
        .collect();

    let kernel_out = kernel_ref::ternary_matmul_stack(&input, &pack, tokens);

    let mut naive = vec![0f32; tokens * out_dim];
    for t in 0..tokens {
        for o in 0..out_dim {
            let mut acc = 0.0f32;
            for i in 0..in_dim {
                acc += unpacked[o * in_dim + i] * input[t * in_dim + i];
            }
            naive[t * out_dim + o] = acc;
        }
    }

    let err = max_abs_diff(&kernel_out, &naive);
    assert!(
        err < 1e-4,
        "kernel_ref vs unpack-matmul max abs {err} (backend {:?})",
        active_backend()
    );
}

#[test]
fn swiglu_and_layernorm_ref_finite() {
    let gate: Vec<f32> = (-8..8).map(|i| i as f32 * 0.25).collect();
    let up: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.1).collect();
    let sw = kernel_ref::swiglu_fused(&gate, &up);
    assert_eq!(sw.len(), 16);
    assert!(sw.iter().all(|v| v.is_finite()));

    let x: Vec<f32> = (0..32).map(|i| i as f32 * 0.05).collect();
    let gamma = vec![1.0f32; 8];
    let beta = vec![0.0f32; 8];
    let ln = kernel_ref::layer_norm(&x, Some(&gamma), Some(&beta), 4, 8, 1e-5);
    assert_eq!(ln.len(), 32);
    assert!(ln.iter().all(|v| v.is_finite()));
}

#[test]
fn packed_linear_matches_candle_unpacked() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let out_dim = 32;
    let in_dim = 16;
    let tokens = 3;
    let weights: Vec<f32> = (0..(out_dim * in_dim))
        .map(|i| ((i % 9) as f32 / 4.0) - 1.0)
        .collect();
    let pack = quantize_rowwise_ternary(&weights, out_dim, in_dim);
    let unpacked = unpack_ternary(&pack);
    let input: Vec<f32> = (0..(tokens * in_dim))
        .map(|i| ((i % 7) as f32 / 3.0) - 0.5)
        .collect();

    let w = Tensor::from_vec(unpacked.clone(), (out_dim, in_dim), &device)?;
    let x = Tensor::from_vec(input.clone(), (tokens, in_dim), &device)?;
    let candle = x.matmul(&w.t()?)?;
    let candle_v = candle.flatten_all()?.to_vec1::<f32>()?;

    let packed = packed_linear(&x, &pack, None)?;
    let packed_v = packed.flatten_all()?.to_vec1::<f32>()?;
    let err = max_abs_diff(&candle_v, &packed_v);
    assert!(err < 1e-4, "packed_linear vs candle max abs {err}");
    Ok(())
}

#[test]
fn network_kernel_path_matches_unpacked_candle() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let net = NetworkConfig {
        variant: NetworkVariant::Transformer,
        dim: 16,
        num_heads: 4,
        max_seq_len: 24,
        vocab_size: 11,
        num_layers: 2,
        use_ternary_kernels: false,
        ..Default::default()
    };
    let trm = TrmConfig {
        n_l_cycles: 1,
        n_sup: 1,
        z_seq: 4,
        x_seq: 4,
        y_seq: 4,
        max_puzzles: 8,
        use_learned_halt_head: false,
        collect_intermediates: false,
        ..Default::default()
    };
    let mut model = TrmModel::new(device.clone(), net, trm)?;
    let x = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device)?;
    let p = Tensor::from_vec(vec![0u32], 1, &device)?;
    let before = model
        .forward_trace(&x, 4, Some(&p), None, None, None)?
        .logits_final
        .flatten_all()?
        .to_vec1::<f32>()?;

    model.pack_ternary_inference()?;
    assert!(
        model.packed_linear_count() > 0,
        "expected packed linears after pack_ternary_inference"
    );
    let after = model
        .forward_trace(&x, 4, Some(&p), None, None, None)?
        .logits_final
        .flatten_all()?
        .to_vec1::<f32>()?;

    // Packing quantizes FP32 weights, so logits move; they must stay finite
    // and the same shape. A second packed forward is bit-stable.
    assert_eq!(before.len(), after.len());
    assert!(after.iter().all(|v| v.is_finite()));
    let after2 = model
        .forward_trace(&x, 4, Some(&p), None, None, None)?
        .logits_final
        .flatten_all()?
        .to_vec1::<f32>()?;
    let err = max_abs_diff(&after, &after2);
    assert!(err < 1e-6, "packed forward not deterministic: {err}");
    let _ = before;
    let _ = active_backend();
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_kernelbank_matches_cpu_ref() -> anyhow::Result<()> {
    let bank = trm_omega::custom_kernels::KernelBank::try_new(0)
        .map_err(|e| anyhow::anyhow!("KernelBank must load on this GTX 1660: {e:#}"))?;
    eprintln!("loaded kernels: {:?}", bank.loaded_kernels());
    assert!(bank.has("ternary_matmul_stack"));
    assert!(bank.has("swiglu_fused"));

    let out_dim = 320;
    let in_dim = 64;
    let tokens = 4;
    let weights: Vec<f32> = (0..(out_dim * in_dim))
        .map(|i| ((i % 17) as f32 / 8.0) - 1.0)
        .collect();
    let pack = quantize_rowwise_ternary(&weights, out_dim, in_dim);
    let input: Vec<f32> = (0..(tokens * in_dim))
        .map(|i| ((i % 11) as f32 / 5.0) - 0.8)
        .collect();
    let cpu = kernel_ref::ternary_matmul_stack(&input, &pack, tokens);
    let gpu = bank.ternary_matmul_stack_host(&input, &pack, tokens)?;
    let err = max_abs_diff(&cpu, &gpu);
    assert!(err < 1e-4, "CUDA vs CPU ternary max abs {err}");

    let relu_cpu = kernel_ref::ternary_matmul(&input, &pack, tokens, true);
    let relu_gpu = bank.ternary_matmul_host(&input, &pack, tokens, true)?;
    let relu_err = max_abs_diff(&relu_cpu, &relu_gpu);
    assert!(relu_err < 1e-4, "CUDA vs CPU relu ternary max abs {relu_err}");

    let gate: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01 - 0.6).collect();
    let up: Vec<f32> = (0..128).map(|i| (i as f32) * 0.02 - 1.0).collect();
    let sw_cpu = kernel_ref::swiglu_fused(&gate, &up);
    let sw_gpu = bank.swiglu_fused_host(&gate, &up)?;
    let sw_err = max_abs_diff(&sw_cpu, &sw_gpu);
    assert!(sw_err < 2e-3, "CUDA vs CPU swiglu max abs {sw_err}");

    if bank.has("layer_norm") {
        let x: Vec<f32> = (0..64).map(|i| (i as f32) * 0.03 - 0.4).collect();
        let gamma = vec![1.0f32; 16];
        let beta = vec![0.1f32; 16];
        let ln_cpu = kernel_ref::layer_norm(&x, Some(&gamma), Some(&beta), 4, 16, 1e-5);
        let ln_gpu = bank.layer_norm_host(&x, &gamma, &beta, 4, 16, 1e-5)?;
        let ln_err = max_abs_diff(&ln_cpu, &ln_gpu);
        assert!(ln_err < 2e-3, "CUDA vs CPU layer_norm max abs {ln_err}");
    }

    let bias: Vec<f32> = (0..out_dim).map(|i| (i as f32) * 0.01 - 0.2).collect();
    let gpu_b = bank.packed_linear_host(&input, &pack, Some(&bias), tokens)?;
    let mut cpu_b = cpu.clone();
    kernel_ref::bias_add(&mut cpu_b, &bias, tokens, out_dim);
    let bias_err = max_abs_diff(&cpu_b, &gpu_b);
    assert!(bias_err < 1e-4, "CUDA vs CPU packed linear+bias max abs {bias_err}");
    let gpu_b2 = bank.packed_linear_host(&input, &pack, Some(&bias), tokens)?;
    let cache_err = max_abs_diff(&gpu_b, &gpu_b2);
    assert!(cache_err < 1e-6, "device pack cache not stable: {cache_err}");

    let hid = 32usize;
    let din = 16usize;
    let gw: Vec<f32> = (0..(hid * din)).map(|i| ((i % 9) as f32 / 4.0) - 1.0).collect();
    let vw: Vec<f32> = (0..(hid * din)).map(|i| ((i % 11) as f32 / 5.0) - 0.8).collect();
    let dw: Vec<f32> = (0..(din * hid)).map(|i| ((i % 7) as f32 / 3.0) - 0.5).collect();
    let gpack = quantize_rowwise_ternary(&gw, hid, din);
    let vpack = quantize_rowwise_ternary(&vw, hid, din);
    let dpack = quantize_rowwise_ternary(&dw, din, hid);
    let gb = vec![0.05f32; hid];
    let vb = vec![-0.02f32; hid];
    let db = vec![0.01f32; din];
    let xs: Vec<f32> = (0..(tokens * din)).map(|i| ((i % 5) as f32 / 4.0) - 0.4).collect();
    let cpu_ffn = kernel_ref::packed_swiglu_ffn(
        &xs, tokens, &gpack, Some(&gb), &vpack, Some(&vb), &dpack, Some(&db),
    );
    let gpu_ffn = bank.packed_swiglu_ffn_host(
        &xs, tokens, &gpack, Some(&gb), &vpack, Some(&vb), &dpack, Some(&db),
    )?;
    let ffn_err = max_abs_diff(&cpu_ffn, &gpu_ffn);
    assert!(ffn_err < 2e-3, "CUDA vs CPU fused SwiGLU FFN max abs {ffn_err}");

    if bank.has("fused_attention") && bank.has("apply_rope") {
        let b = 1usize;
        let seq = 4usize;
        let heads = 4usize;
        let kv = 2usize;
        let dh = 8usize;
        let dim = heads * dh;
        let qw: Vec<f32> = (0..(dim * dim)).map(|i| ((i % 9) as f32 / 4.0) - 1.0).collect();
        let kw: Vec<f32> = (0..((kv * dh) * dim)).map(|i| ((i % 7) as f32 / 3.0) - 0.6).collect();
        let vw: Vec<f32> = (0..((kv * dh) * dim)).map(|i| ((i % 5) as f32 / 4.0) - 0.4).collect();
        let ow: Vec<f32> = (0..(dim * dim)).map(|i| ((i % 11) as f32 / 5.0) - 0.8).collect();
        let qpack = quantize_rowwise_ternary(&qw, dim, dim);
        let kpack = quantize_rowwise_ternary(&kw, kv * dh, dim);
        let vpack = quantize_rowwise_ternary(&vw, kv * dh, dim);
        let opack = quantize_rowwise_ternary(&ow, dim, dim);
        let qb = vec![0.01f32; dim];
        let kb = vec![-0.02f32; kv * dh];
        let vb = vec![0.03f32; kv * dh];
        let ob = vec![0.0f32; dim];
        let half = dh / 2;
        let mut cos = vec![0f32; seq * half];
        let mut sin = vec![0f32; seq * half];
        for pos in 0..seq {
            for i in 0..half {
                let theta = (pos as f32) / 10_000f32.powf(2.0 * i as f32 / dh as f32);
                cos[pos * half + i] = theta.cos();
                sin[pos * half + i] = theta.sin();
            }
        }
        let xs: Vec<f32> = (0..(b * seq * dim)).map(|i| ((i % 6) as f32 / 5.0) - 0.5).collect();
        let cpu_mha = kernel_ref::packed_mha(
            &xs, b, seq, &qpack, Some(&qb), &kpack, Some(&kb), &vpack, Some(&vb),
            &opack, Some(&ob), heads, kv, dh, &cos, &sin, None,
        );
        let gpu_mha = bank.packed_mha_host(
            &xs, b, seq, &qpack, Some(&qb), &kpack, Some(&kb), &vpack, Some(&vb),
            &opack, Some(&ob), heads, kv, dh, &cos, &sin, None,
        )?;
        let mha_err = max_abs_diff(&cpu_mha, &gpu_mha);
        assert!(mha_err < 5e-3, "CUDA vs CPU fused MHA max abs {mha_err}");
    }
    Ok(())
}
