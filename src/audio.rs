use std::f32::consts::PI;
use ndarray::{Array1, Array2};
use rustfft::{FftPlanner, num_complex::Complex};
use rubato::{Resampler, SincFixedIn, SincInterpolationType, SincInterpolationParameters, WindowFunction};
use hound::WavReader;
use anyhow::{Result, Context};

// Whisper parameters
const SAMPLE_RATE: usize = 16000;
const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
const CHUNK_LENGTH: usize = 30;
const N_SAMPLES: usize = CHUNK_LENGTH * SAMPLE_RATE;
const N_MELS: usize = 80;

pub struct AudioProcessor {
    mel_filters: Array2<f32>,
}

impl AudioProcessor {
    pub fn new() -> Result<Self> {
        // Generate Mel filters
        let mel_filters = mel_filter_bank(
            SAMPLE_RATE as f32,
            N_FFT as f32,
            N_MELS,
            0.0,
            8000.0,
        );
        Ok(Self { mel_filters })
    }

    pub fn load_and_preprocess(&self, path: &str) -> Result<Array2<f32>> {
        println!("Loading audio from: {}", path);
        let (samples, sr) = read_wav(path)?;
        // Just call process_pcm with the samples and their original sample rate
        // We do the resampling here if needed because process_pcm expects input ready for log_mel_spectrogram?
        // Wait, log_mel_spectrogram expects 16kHz.
        // So I should do resampling in process_pcm or before?
        // load_and_preprocess used to do resampling before log_mel_spectrogram.
        // I will make process_pcm take &[f32] which ARE already 16kHz, OR make it take SR?
        // To be flexible for streaming (which is usually 16kHz), let's assume process_pcm takes 16kHz.
        // But better to be explicit.
        
        let resampled = if sr != SAMPLE_RATE {
            resample_audio(&samples, sr, SAMPLE_RATE)?
        } else {
            samples
        };
        
        Ok(self.process_pcm(&resampled))
    }

    /// Process PCM audio samples (must be 16kHz).
    pub fn process_pcm(&self, samples: &[f32]) -> Array2<f32> {
        self.log_mel_spectrogram(samples)
    }

    fn log_mel_spectrogram(&self, audio: &[f32]) -> Array2<f32> {
        // Pad audio to N_SAMPLES if needed, or slice it. 
        let mut padded_audio = audio.to_vec();
        if padded_audio.len() < N_SAMPLES {
            padded_audio.resize(N_SAMPLES, 0.0);
        } else if padded_audio.len() > N_SAMPLES {
            padded_audio.truncate(N_SAMPLES);
        }

        // STFT
        let window = hann_window(N_FFT);
        let stft_out = stft(&padded_audio, N_FFT, HOP_LENGTH, &window);
        
        // Magnitudes squared
        let magnitudes = stft_out.mapv(|c| c.norm_sqr());
        // Apply only up to N_FFT/2 + 1
        let magnitudes = magnitudes.slice(ndarray::s![.., ..(N_FFT / 2 + 1)]);

        // Mel transform: [Time, Freq] x [Freq, Mel] -> [Time, Mel]
        let mel_spec = magnitudes.dot(&self.mel_filters);
        
        // Log10
        let mut log_spec = mel_spec.mapv(|x| (x.max(1e-10)).log10());

        // Standard scaling for Whisper:
        // log_spec = (log_spec + 4.0) / 4.0
        let max_val = log_spec.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        log_spec.mapv_inplace(|x| (x.max(max_val - 8.0) + 4.0) / 4.0);

        // Ensure exactly 3000 frames
        let current_frames = log_spec.shape()[0];
        let target_frames = 3000;
        
        let final_spec = if current_frames < target_frames {
            let mut padded = Array2::zeros((target_frames, N_MELS));
            padded.slice_mut(ndarray::s![..current_frames, ..]).assign(&log_spec);
            padded
        } else if current_frames > target_frames {
            log_spec.slice(ndarray::s![..target_frames, ..]).to_owned()
        } else {
            log_spec
        };

        // Transpose to [Mel, Time] -> [80, 3000]
        final_spec.t().to_owned()
    }
}

fn read_wav(path: &str) -> Result<(Vec<f32>, usize)> {
    let mut reader = WavReader::open(path).with_context(|| format!("Failed to open wav file '{}'", path))?;
    let spec = reader.spec();
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.map(|x| x as f32 / 32768.0))
        .collect::<Result<Vec<f32>, _>>()?;
    Ok((samples, spec.sample_rate as usize))
}

fn resample_audio(samples: &[f32], from_sr: usize, to_sr: usize) -> Result<Vec<f32>> {
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    
    let ratio = to_sr as f64 / from_sr as f64;
    let mut resampler = SincFixedIn::<f32>::new(
        ratio,
        ratio, 
        params,
        samples.len(),
        1,
    )?;

    let waves_in = vec![samples.to_vec()];
    let waves_out = resampler.process(&waves_in, None)?;
    Ok(waves_out[0].clone())
}

fn hann_window(size: usize) -> Array1<f32> {
    Array1::from_shape_fn(size, |i| {
        0.5 * (1.0 - (2.0 * PI * i as f32 / size as f32).cos())
    })
}

fn stft(input: &[f32], n_fft: usize, hop_length: usize, window: &Array1<f32>) -> Array2<Complex<f32>> {
    let n_frames = (input.len() - n_fft) / hop_length + 1;
    let mut output = Array2::<Complex<f32>>::zeros((n_frames, n_fft));
    
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);

    for (i, row) in output.outer_iter_mut().enumerate() {
        let start = i * hop_length;
        let end = start + n_fft;
        let mut frame = input[start..end].to_vec();
        
        // Apply window
        for (j, val) in frame.iter_mut().enumerate() {
            *val *= window[j];
        }

        let mut complex_frame: Vec<Complex<f32>> = frame.iter().map(|&x| Complex::new(x, 0.0)).collect();
        fft.process(&mut complex_frame);
        
        let mut row_view = row;
        for (j, c) in complex_frame.iter().enumerate() {
            row_view[j] = *c;
        }
    }
    output
}

// Minimal Mel Filterbank implementation (approximation)
fn mel_filter_bank(sr: f32, n_fft: f32, n_mels: usize, fmin: f32, fmax: f32) -> Array2<f32> {
    let fft_freqs = (0..(n_fft as usize / 2 + 1)).map(|i| i as f32 * sr / n_fft).collect::<Vec<_>>();
    let n_freqs = fft_freqs.len();

    let mel_min = 2595.0 * (1.0 + fmin / 700.0).log10();
    let mel_max = 2595.0 * (1.0 + fmax / 700.0).log10();
    
    let mels = (0..(n_mels + 2)).map(|i| {
        let m = mel_min + (mel_max - mel_min) * i as f32 / (n_mels as f32 + 1.0);
        700.0 * (10.0f32.powf(m / 2595.0) - 1.0)
    }).collect::<Vec<_>>();

    let mut weights = Array2::<f32>::zeros((n_freqs, n_mels));

    for i in 0..n_mels {
        let f_prev = mels[i];
        let f_curr = mels[i + 1];
        let f_next = mels[i + 2];

        for (j, &freq) in fft_freqs.iter().enumerate() {
            if freq >= f_prev && freq < f_curr {
                 weights[[j, i]] = (freq - f_prev) / (f_curr - f_prev);
            } else if freq >= f_curr && freq < f_next {
                 weights[[j, i]] = (f_next - freq) / (f_next - f_curr);
            }
        }
    }

    // Slaney-style normalization
    for i in 0..n_mels {
        let width = mels[i + 2] - mels[i];
        let norm_factor = 2.0 / width;
        
        let mut col = weights.slice_mut(ndarray::s![.., i]);
        col.mapv_inplace(|x| x * norm_factor);
    }
    
    weights
}
