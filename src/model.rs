use std::sync::Mutex;
use std::collections::HashMap;
use anyhow::{Result, anyhow};
use ort::{
    session::{Session, builder::GraphOptimizationLevel, SessionInputValue},
    value::Tensor,
};
use ndarray::{Array1, Array2, Array3, Array4, Axis};

const SOT: i64 = 50258;
const EOT: i64 = 50257;
const MAX_LEN: usize = 448;
const N_LAYER: usize = 32;
const D_MODEL: usize = 1280;

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
        // === 1. Encoder ===
        let batch_mel = mel.view().insert_axis(Axis(0));
        
        let (cross_k, cross_v) = {
            let inputs = ort::inputs![
                "mel" => Tensor::from_array(batch_mel.to_owned())?,
            ];

            let mut encoder_session = self.encoder.lock().map_err(|e| anyhow!("Failed to lock encoder: {}", e))?;
            let encoder_out = encoder_session.run(inputs)?;
            
            // Helper to convert output to owned Tensor
            fn extract_to_tensor(out: &ort::value::DynValue) -> Result<Tensor<f32>> {
                let (shape, data) = out.try_extract_tensor::<f32>()?;
                let shape_vec: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
                // We know it is 4D [32, 1, 1500, 1280] usually
                let array = Array4::from_shape_vec(
                    (shape_vec[0], shape_vec[1], shape_vec[2], shape_vec[3]),
                    data.to_vec()
                )?;
                Ok(Tensor::from_array(array)?)
            }

            let k_owned = extract_to_tensor(&encoder_out["n_layer_cross_k"])?;
            let v_owned = extract_to_tensor(&encoder_out["n_layer_cross_v"])?;
            
            (k_owned, v_owned)
        };

        // === 2. Decoder Loop (Greedy) ===
        let mut tokens = vec![SOT]; 
        
        let mut decoder_session = self.decoder.lock().map_err(|e| anyhow!("Failed to lock decoder: {}", e))?;

        let mut self_k_cache = Array4::<f32>::zeros((N_LAYER, 1, MAX_LEN, D_MODEL));
        let mut self_v_cache = Array4::<f32>::zeros((N_LAYER, 1, MAX_LEN, D_MODEL));

        for i in 0..MAX_LEN {
            let offset = i as i64;
            let current_token = *tokens.last().unwrap();
            let token_input = Array2::from_shape_vec((1, 1), vec![current_token])?;
            let offset_input = Array1::from_shape_vec((1,), vec![offset])?;

            let mut inputs: HashMap<String, SessionInputValue<'_>> = HashMap::new();
            inputs.insert("tokens".to_string(), Tensor::from_array(token_input)?.into());
            inputs.insert("in_n_layer_self_k_cache".to_string(), Tensor::from_array(self_k_cache.clone())?.into());
            inputs.insert("in_n_layer_self_v_cache".to_string(), Tensor::from_array(self_v_cache.clone())?.into());
            inputs.insert("n_layer_cross_k".to_string(), cross_k.clone().into());
            inputs.insert("n_layer_cross_v".to_string(), cross_v.clone().into());
            inputs.insert("offset".to_string(), Tensor::from_array(offset_input)?.into());

            let outputs = match decoder_session.run(inputs) {
                Ok(o) => o,
                Err(e) => return Err(anyhow!("Decoder run failed at step {}: {}", i, e)),
            };

            // Process outputs
            {
                let (shape, data) = outputs["out_n_layer_self_k_cache"].try_extract_tensor::<f32>()?;
                // Should match [32, 1, 448, 1280]
                let shape_vec: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
                let out_arr = Array4::from_shape_vec(
                    (shape_vec[0], shape_vec[1], shape_vec[2], shape_vec[3]),
                    data.to_vec()
                )?;
                self_k_cache.assign(&out_arr);
            }
            {
                let (shape, data) = outputs["out_n_layer_self_v_cache"].try_extract_tensor::<f32>()?;
                let shape_vec: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
                let out_arr = Array4::from_shape_vec(
                    (shape_vec[0], shape_vec[1], shape_vec[2], shape_vec[3]),
                    data.to_vec()
                )?;
                self_v_cache.assign(&out_arr);
            }

            let next_token = {
                let (shape, data) = outputs["logits"].try_extract_tensor::<f32>()?;
                // Shape [1, 1, Vocab]
                let shape_vec: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
                let logits_arr = Array3::from_shape_vec(
                    (shape_vec[0], shape_vec[1], shape_vec[2]),
                    data.to_vec()
                )?;
                
                let logits_slice = logits_arr.slice(ndarray::s![0, 0, ..]);
                let (token, _) = logits_slice.iter().enumerate().fold(
                    (0, f32::NEG_INFINITY), 
                    |(argmax, max), (i, &val)| if val > max { (i, val) } else { (argmax, max) }
                );
                token as i64
            };

            if next_token == EOT {
                break;
            }
            tokens.push(next_token);
        }

        Ok(tokens)
    }
}
