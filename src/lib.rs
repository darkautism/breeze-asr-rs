pub mod audio;
pub mod model;
pub mod tokenizer;

use std::path::Path;
use anyhow::Result;
use hf_hub::api::sync::Api;

use crate::audio::AudioProcessor;
use crate::model::BreezeModel;
use crate::tokenizer::Tokenizer;

pub struct BreezeASR {
    model: BreezeModel,
    tokenizer: Tokenizer,
    audio_processor: AudioProcessor,
}

impl BreezeASR {
    /// Initialize the model.
    /// If `model_dir` is provided, loads from there.
    /// Otherwise, downloads from Hugging Face.
    pub fn init(model_dir: Option<&str>) -> Result<Self> {
        let (encoder_path, decoder_path, tokenizer_path) = match model_dir {
            Some(dir) => {
                let dir = Path::new(dir);
                (
                    dir.join("breeze-asr-25-encoder.onnx"),
                    dir.join("breeze-asr-25-decoder.onnx"),
                    dir.join("breeze-asr-25-tokens.txt"),
                )
            }
            None => {
                let api = Api::new()?;
                let repo = api.model("MediaTek-Research/Breeze-ASR-25-onnx-250806".to_string());
                (
                    repo.get("breeze-asr-25-encoder.onnx")?,
                    repo.get("breeze-asr-25-decoder.onnx")?,
                    repo.get("breeze-asr-25-tokens.txt")?,
                )
            }
        };

        if model_dir.is_none() {
            let api = Api::new()?;
            let repo = api.model("MediaTek-Research/Breeze-ASR-25-onnx-250806".to_string());
            let _ = repo.get("breeze-asr-25-encoder.weights")?;
            let _ = repo.get("breeze-asr-25-decoder.weights")?;
        }

        let model = BreezeModel::new(
            encoder_path.to_str().unwrap(), 
            decoder_path.to_str().unwrap()
        )?;
        
        let tokenizer = Tokenizer::new(tokenizer_path.to_str().unwrap())?;
        let audio_processor = AudioProcessor::new()?;

        Ok(Self {
            model,
            tokenizer,
            audio_processor,
        })
    }

    pub fn infer_file(&self, path: &str) -> Result<Vec<String>> {
        let mel = self.audio_processor.load_and_preprocess(path)?;
        let tokens = self.model.infer(&mel)?;
        let text = self.tokenizer.decode(&tokens);

        Ok(vec![text])
    }
}
