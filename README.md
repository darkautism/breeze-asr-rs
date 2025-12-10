# Breeze ASR Rs

這是一個Rust版本的Breeze ASR

關於該ASR的詳細性能請見：[Breeze-ASR-25](https://github.com/mtkresearch/Breeze-ASR-25)

## Useage


請見examples

```
use breeze_asr_rs::BreezeASR;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <wav_file>", args[0]);
        return Ok(());
    }
    
    let audio_path = &args[1];

    println!("Initializing Breeze ASR...");
    // Pass None to download from HF, or Some("/path/to/models") if local
    let asr = BreezeASR::init(None)?;

    println!("Inferring...");
    let result = asr.infer_file(audio_path)?;

    for vt in result {
        println!("Content: {}", vt);
    }

    Ok(())
}
```