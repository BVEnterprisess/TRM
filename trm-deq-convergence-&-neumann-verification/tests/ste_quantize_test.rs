#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};
    use trm_omega::quantize::{
        quantize_rowwise_ternary, relative_l2_error, ternary_ste, unpack_ternary,
    };

    #[test]
    fn test_ternary_ste_quantization_values() -> candle_core::Result<()> {
        let dev = Device::Cpu;
        let w_data = vec![-1.2f32, -0.6, -0.1, 0.0, 0.1, 0.6, 1.4, -0.8];
        let w = Tensor::from_vec(w_data.clone(), (2, 4), &dev)?;
        let q = ternary_ste(&w)?;
        let q_vec = q.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(q_vec.len(), 8);
        for &val in &q_vec {
            assert!(val.is_finite());
        }

        // Per-row values must be alpha * {-1, 0, +1}.
        for row in 0..2 {
            let row_w = &w_data[row * 4..(row + 1) * 4];
            let alpha = row_w.iter().map(|v| v.abs()).sum::<f32>() / 4.0;
            for j in 0..4 {
                let qv = q_vec[row * 4 + j];
                let level = qv / alpha.max(1e-8);
                let nearest = if level <= -0.5 {
                    -1.0
                } else if level >= 0.5 {
                    1.0
                } else {
                    0.0
                };
                assert!(
                    (level - nearest).abs() < 1e-5,
                    "row {row} col {j}: STE value {qv} / alpha {alpha} = {level}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn pack_unpack_roundtrip_and_l2_bound() {
        let out_dim = 8;
        let in_dim = 16;
        let weights: Vec<f32> = (0..(out_dim * in_dim))
            .map(|i| ((i % 13) as f32 / 6.0) - 1.0)
            .collect();
        let packed = quantize_rowwise_ternary(&weights, out_dim, in_dim);
        let unpacked = unpack_ternary(&packed);
        assert_eq!(unpacked.len(), weights.len());
        assert_eq!(packed.alphas.len(), out_dim);

        let err = relative_l2_error(&weights, &unpacked);
        assert!(err.is_finite());
        assert!(
            err < 0.85,
            "row-wise ternary relative L2 {err} is unreasonably high"
        );

        for o in 0..out_dim {
            let alpha = packed.alphas[o];
            for j in 0..in_dim {
                let v = unpacked[o * in_dim + j];
                let level = v / alpha.max(1e-8);
                let ok = (level - (-1.0)).abs() < 1e-5
                    || level.abs() < 1e-5
                    || (level - 1.0).abs() < 1e-5;
                assert!(ok, "unpacked[{o},{j}]={v} is not alpha {alpha} times ternary");
            }
        }
    }
}
