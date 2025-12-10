#[cfg(feature = "stream")]
use std::collections::VecDeque;
#[cfg(feature = "stream")]
use voice_activity_detector::{IteratorExt, VoiceActivityDetector};

#[cfg(feature = "stream")]
pub const CHUNK_SIZE: usize = 512;

#[cfg(feature = "stream")]
#[derive(Debug, Clone, Copy)]
pub struct VadConfig {
    pub sample_rate: u32,
    pub speech_threshold: f32,
    pub silence_duration_ms: u32,
    pub max_speech_duration_ms: u32,
    pub rollback_duration_ms: u32,
    pub min_speech_duration_ms: u32,
    pub notify_silence_after_ms: Option<u32>,
}

#[cfg(feature = "stream")]
impl Default for VadConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            speech_threshold: 0.5,
            silence_duration_ms: 500,
            max_speech_duration_ms: 10000,
            rollback_duration_ms: 200,
            min_speech_duration_ms: 250,
            notify_silence_after_ms: None,
        }
    }
}

#[cfg(feature = "stream")]
#[derive(Debug)]
enum VadState {
    Waiting,
    Recording,
}

#[cfg(feature = "stream")]
#[derive(Debug)]
pub enum VadOutput {
    Segment(Vec<i16>),
    SilenceNotification,
}

#[cfg(feature = "stream")]
#[derive(Debug)]
pub struct VadProcessor {
    vad: VoiceActivityDetector,
    config: VadConfig,
    state: VadState,
    current_segment: Vec<i16>,
    history_buffer: VecDeque<i16>,
    silence_chunks: u32,
    speech_chunks: u32,
    waiting_dropped_chunks: u32,
    notified_silence: bool,
}

#[cfg(feature = "stream")]
impl VadProcessor {
    pub fn new(config: VadConfig) -> anyhow::Result<Self> {
        let vad = VoiceActivityDetector::builder()
            .sample_rate(config.sample_rate)
            .chunk_size(CHUNK_SIZE)
            .build()
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(Self {
            vad,
            config,
            state: VadState::Waiting,
            current_segment: Vec::new(),
            history_buffer: VecDeque::new(),
            silence_chunks: 0,
            speech_chunks: 0,
            waiting_dropped_chunks: 0,
            notified_silence: false,
        })
    }

    pub fn set_notify_silence_after_ms(&mut self, ms: Option<u32>) {
        self.config.notify_silence_after_ms = ms;
        if ms.is_none() {
            self.notified_silence = false;
        }
    }

    pub fn process_chunk(&mut self, chunk: &[i16; CHUNK_SIZE]) -> Option<VadOutput> {
        let chunk_duration_ms = (CHUNK_SIZE as f32 / self.config.sample_rate as f32) * 1000.0;
        let probability = chunk
            .iter()
            .copied()
            .predict(&mut self.vad)
            .next()
            .unwrap()
            .1;

        match self.state {
            VadState::Waiting => {
                self.history_buffer.extend(chunk.iter().copied());

                let rollback_samples = ((self.config.rollback_duration_ms as f32 / 1000.0)
                    * self.config.sample_rate as f32) as usize;
                while self.history_buffer.len() > rollback_samples {
                    self.history_buffer.pop_front();
                }

                if probability > self.config.speech_threshold {
                    self.state = VadState::Recording;
                    self.current_segment.extend(self.history_buffer.iter());
                    self.history_buffer.clear();
                    self.silence_chunks = 0;
                    self.speech_chunks = 0;
                    self.waiting_dropped_chunks = 0;
                    self.notified_silence = false;
                } else {
                    if let Some(limit_ms) = self.config.notify_silence_after_ms {
                        self.waiting_dropped_chunks += 1;
                        let dropped_duration = self.waiting_dropped_chunks as f32 * chunk_duration_ms;
                        if dropped_duration >= limit_ms as f32 && !self.notified_silence {
                            self.notified_silence = true;
                            return Some(VadOutput::SilenceNotification);
                        }
                    }
                }
                None
            }
            VadState::Recording => {
                self.current_segment.extend(chunk);
                self.speech_chunks += 1;

                if probability > self.config.speech_threshold {
                    self.silence_chunks = 0;
                    let speech_duration_ms = self.speech_chunks as f32 * chunk_duration_ms;
                    if speech_duration_ms >= self.config.max_speech_duration_ms as f32 {
                        return self.finalize_segment(false);
                    }
                } else {
                    self.silence_chunks += 1;
                    let silence_duration_ms = self.silence_chunks as f32 * chunk_duration_ms;
                    if silence_duration_ms >= self.config.silence_duration_ms as f32 {
                        return self.finalize_segment(true);
                    }
                }
                None
            }
        }
    }

    fn finalize_segment(&mut self, trim_tail: bool) -> Option<VadOutput> {
        if self.current_segment.is_empty() {
            self.reset();
            return None;
        }

        let mut segment = if trim_tail {
            let chunk_len = CHUNK_SIZE;
            let silence_len = (self.silence_chunks as usize) * chunk_len;
            let valid_len = self.current_segment.len().saturating_sub(silence_len);
            if valid_len == 0 {
                Vec::new()
            } else {
                self.current_segment[..valid_len].to_vec()
            }
        } else {
            self.current_segment.clone()
        };

        let duration_ms =
            (segment.len() as f32 / self.config.sample_rate as f32) * 1000.0;
        if duration_ms < self.config.min_speech_duration_ms as f32 {
            segment.clear();
        }

        self.reset();

        if segment.is_empty() {
            None
        } else {
            Some(VadOutput::Segment(segment))
        }
    }

    fn reset(&mut self) {
        self.current_segment.clear();
        self.history_buffer.clear();
        self.silence_chunks = 0;
        self.speech_chunks = 0;
        self.state = VadState::Waiting;
        self.waiting_dropped_chunks = 0;
        self.notified_silence = false;
    }

    pub fn finish(&mut self) -> Option<VadOutput> {
        if !self.current_segment.is_empty() {
             let duration_ms = (self.current_segment.len() as f32 / self.config.sample_rate as f32) * 1000.0;
             if duration_ms < self.config.min_speech_duration_ms as f32 {
                 self.reset();
                 return None;
             }

            let segment = self.current_segment.clone();
            self.reset();
            Some(VadOutput::Segment(segment))
        } else {
            None
        }
    }
}
