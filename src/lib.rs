pub mod audio;
pub mod model;
pub mod tokenizer;
#[cfg(feature = "stream")]
pub mod vad;

#[cfg(feature = "stream")]
pub use vad::VadConfig;

use std::path::Path;
use anyhow::Result;
use hf_hub::api::sync::Api;

use crate::audio::AudioProcessor;
use crate::model::BreezeModel;
use crate::tokenizer::Tokenizer;

#[cfg(feature = "stream")]
use crate::vad::{VadProcessor, VadOutput, CHUNK_SIZE};
#[cfg(feature = "stream")]
use async_stream::stream;
#[cfg(feature = "stream")]
use futures::stream::Stream;
#[cfg(feature = "stream")]
use futures::StreamExt;
#[cfg(feature = "stream")]
use std::sync::Mutex;

pub struct BreezeASR {
    model: BreezeModel,
    tokenizer: Tokenizer,
    audio_processor: AudioProcessor,
    #[cfg(feature = "stream")]
    vad_processor: Mutex<Option<VadProcessor>>,
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

        #[cfg(feature = "stream")]
        let vad_processor = Mutex::new(Some(VadProcessor::new(VadConfig::default())?));

        Ok(Self {
            model,
            tokenizer,
            audio_processor,
            #[cfg(feature = "stream")]
            vad_processor,
        })
    }

    /// Initialize with custom VAD configuration.
    #[cfg(feature = "stream")]
    pub fn init_with_vad(model_dir: Option<&str>, vad_config: VadConfig) -> Result<Self> {
        let slf = Self::init(model_dir)?;
        *slf.vad_processor.lock().unwrap() = Some(VadProcessor::new(vad_config)?);
        Ok(slf)
    }

    pub fn infer_file(&self, path: &str) -> Result<Vec<String>> {
        let mel = self.audio_processor.load_and_preprocess(path)?;
        let tokens = self.model.infer(&mel)?;
        let text = self.tokenizer.decode(&tokens);

        Ok(vec![text])
    }

    /// Streaming inference.
    /// Filters out empty or silence-only segments.
    #[cfg(feature = "stream")]
    pub fn infer_stream<'a, S>(
        &'a self,
        input_stream: S,
    ) -> impl Stream<Item = Result<String>> + 'a
    where
        S: Stream<Item = Vec<i16>> + Unpin + 'a,
    {
        stream! {
            let mut stream = input_stream;
            while let Some(chunk) = stream.next().await {
                // Ensure chunk size is CHUNK_SIZE
                if chunk.len() != CHUNK_SIZE {
                    continue; 
                }

                let chunk_arr: &[i16; CHUNK_SIZE] = chunk.as_slice().try_into().unwrap();
                
                let output_opt = {
                    let mut vad_guard = self.vad_processor.lock().unwrap();
                    if let Some(vad) = vad_guard.as_mut() {
                        vad.process_chunk(chunk_arr)
                    } else {
                        None // Should return error maybe?
                    }
                };

                if let Some(output) = output_opt {
                    match output {
                        VadOutput::Segment(segment) => {
                            if let Ok(text) = self.infer_segment(&segment) {
                                if !text.trim().is_empty() {
                                    yield Ok(text);
                                }
                            }
                        },
                        VadOutput::SilenceNotification => {}
                    }
                }
            }
            
            let finish_opt = {
                 let mut vad_guard = self.vad_processor.lock().unwrap();
                 if let Some(vad) = vad_guard.as_mut() {
                     vad.finish()
                 } else {
                     None
                 }
            };

            if let Some(output) = finish_opt {
                 match output {
                    VadOutput::Segment(segment) => {
                        if let Ok(text) = self.infer_segment(&segment) {
                            if !text.trim().is_empty() {
                                yield Ok(text);
                            }
                        }
                    },
                    VadOutput::SilenceNotification => {}
                }
            }
        }
    }

    #[cfg(feature = "stream")]
    fn infer_segment(&self, segment: &[i16]) -> Result<String> {
        // Convert i16 to f32 normalized
        let samples: Vec<f32> = segment.iter().map(|&x| x as f32 / 32768.0).collect();
        
        // Preprocess
        let mel = self.audio_processor.process_pcm(&samples);
        
        // Infer
        let tokens = self.model.infer(&mel)?;
        
        // Decode
        let text = self.tokenizer.decode(&tokens);
        
        Ok(text)
    }
}
