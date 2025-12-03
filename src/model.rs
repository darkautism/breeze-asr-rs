use std::sync::Mutex;
use anyhow::{Result, anyhow};
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor,
};
use ndarray::{Array2, Array3, Axis};

const SOT: i64 = 50258;

pub struct BreezeModel {
    encoder: Mutex<Session>,
    decoder: Mutex<Session>,
}

impl BreezeModel {
    pub fn new(encoder_path: &str, decoder_path: &str) -> Result<Self> {
        let encoder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(encoder_path)?;

        let decoder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(decoder_path)?;

        Ok(Self {
            encoder: Mutex::new(encoder),
            decoder: Mutex::new(decoder)
        })
    }

    pub fn infer(&self, mel: &Array2<f32>) -> Result<Vec<i64>> {
        // 1. Encoder
        let batch_mel = mel.view().insert_axis(Axis(0));

        let enc_tensor = {
            let inputs = ort::inputs![
                "mel" => Tensor::from_array(batch_mel.to_owned())?,
            ];

            let mut encoder_session = self.encoder.lock().map_err(|e| anyhow!("Failed to lock encoder: {}", e))?;
            let encoder_out = encoder_session.run(inputs)?;

            let (enc_shape, enc_data) = encoder_out[0].try_extract_tensor::<f32>()?;
            // Copy data out to a new Tensor that owns its data, decoupling from session
            Tensor::from_array(
                Array3::from_shape_vec(
                    (enc_shape[0] as usize, enc_shape[1] as usize, enc_shape[2] as usize),
                    enc_data.to_vec()
                )?
            )?
            // encoder_out and inputs dropped here
            // encoder_session dropped here (unlocks)
        };

        // 2. Decoder Loop (Greedy)
        let mut tokens = vec![SOT];
        let max_len = 448;

        let mut decoder_session = self.decoder.lock().map_err(|e| anyhow!("Failed to lock decoder: {}", e))?;

        for _ in 0..max_len {
            let input_ids = Array2::from_shape_vec(
                (1, tokens.len()),
                tokens.clone()
            )?;

            let inputs = ort::inputs![
                "input_ids" => Tensor::from_array(input_ids)?,
                "encoder_hidden_states" => enc_tensor.clone(),
            ];

            let outputs = match decoder_session.run(inputs) {
                Ok(o) => o,
                Err(e) => return Err(anyhow!("Decoder run failed: {}. Check if model expects past_key_values.", e)),
            };

            let (out_shape, out_data) = outputs[0].try_extract_tensor::<f32>()?;
            let vocab_size = out_shape[2] as usize;
            let seq_len = out_shape[1] as usize;

            let logits = &out_data[(seq_len - 1) * vocab_size .. seq_len * vocab_size];

            let (next_token, _) = logits.iter().enumerate().fold(
                (0, f32::NEG_INFINITY),
                |(argmax, max), (i, &val)| if val > max { (i, val) } else { (argmax, max) }
            );

            let next_token = next_token as i64;

            if next_token == 50257 { // EOT
                break;
            }

            tokens.push(next_token);
        }

        Ok(tokens)
    }
}
