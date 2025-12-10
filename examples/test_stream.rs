#[cfg(feature = "stream")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use breeze_asr_rs::{BreezeASR, VadConfig};
    use futures::stream::StreamExt;
    use async_stream::stream;
    use std::io::Read;
    use std::time::Duration;
    use hound;
    
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <wav_file>", args[0]);
        return Ok(());
    }
    
    let audio_path = &args[1];
    
    // Check if test file exists, if not, skip (this is just a verification script)
    if !std::path::Path::new(audio_path).exists() {
        println!("Test audio not found, skipping streaming test.");
        return Ok(());
    }

    println!("Initializing BreezeASR with VAD...");
    let mut asr = BreezeASR::init_with_vad(None, VadConfig::default())?;

    println!("Starting stream inference...");
    
    let mut reader = hound::WavReader::open(audio_path)?;
    let samples: Vec<i16> = reader.samples::<i16>().map(|x| x.unwrap()).collect();

    // Chunk size 512
    let chunk_size = 512;
    
    // Pin the input stream to make it Unpin
    let stream = Box::pin(stream! {
        for chunk in samples.chunks(chunk_size) {
            if chunk.len() == chunk_size {
                yield chunk.to_vec();
                // Simulate real-time delay (optional, fast-forward here)
                // tokio::time::sleep(Duration::from_millis(10)).await; 
            }
        }
    });

    let mut output_stream = Box::pin(asr.infer_stream(stream));

    while let Some(result) = output_stream.next().await {
        match result {
            Ok(text) => println!("Stream output: {}", text),
            Err(e) => eprintln!("Error: {}", e),
        }
    }

    Ok(())
}

#[cfg(not(feature = "stream"))]
fn main() {
    println!("Stream feature not enabled.");
}
