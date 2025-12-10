use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use anyhow::Result;
use base64::{Engine as _, engine::general_purpose};

pub struct Tokenizer {
    id_to_bytes: HashMap<i64, Vec<u8>>,
}

impl Tokenizer {
    pub fn new(path: &str) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut id_to_bytes = HashMap::new();

        for (i, line) in reader.lines().enumerate() {
            let line = line?;
            // Format is: BASE64_TOKEN NUMBER
            // Example: IQ== 0
            
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }
            let raw_token = parts[0];

            let decoded_bytes = decode_token_bytes(raw_token);
            id_to_bytes.insert(i as i64, decoded_bytes);
        }

        Ok(Self { id_to_bytes })
    }

    pub fn decode(&self, ids: &[i64]) -> String {
        let mut all_bytes = Vec::new();
        for &id in ids {
            if let Some(bytes) = self.id_to_bytes.get(&id) {
                // Check for special tokens format <|...|>
                // We convert to string temporarily to check, or check bytes directly.
                // Since special tokens are usually ASCII, we can check.
                // But for simplicity/correctness with BPE:
                // Special tokens in Whisper are usually added to the vocab.
                // If it's a special token, we might want to skip it or handle it.
                // For now, simply appending all bytes is the safest way to handle split UTF-8.
                // But we should filter out special tokens like <|startoftranscript|> if they are in the output.
                
                // Heuristic: if it looks like a special token (starts with <|), skip it?
                // Or just let user handle it? The user output had empty strings for special tokens.
                // Breeze tokens might be base64 encoded even for special tokens.
                // Let's inspect: "<|" is "PDx8", "|>" is "fD4=".
                
                // Let's try to detect special tokens by decoding individually *just for checking*, 
                // but use the raw bytes for the final string.
                // Note: This check might be slow but safe.
                if bytes.len() > 4 && bytes.starts_with(b"<|") && bytes.ends_with(b"|>") {
                    continue;
                }
                
                all_bytes.extend_from_slice(bytes);
            }
        }
        String::from_utf8_lossy(&all_bytes).into_owned()
    }
}

fn decode_token_bytes(input: &str) -> Vec<u8> {
    // 1. Try Standard
    if let Ok(bytes) = general_purpose::STANDARD.decode(input) {
        return bytes;
    }
    
    // 2. Try Standard No Pad
    if let Ok(bytes) = general_purpose::STANDARD_NO_PAD.decode(input) {
        return bytes;
    }

    // 3. Manual Padding fix
    let mut padded = input.to_string();
    while padded.len() % 4 != 0 {
        padded.push('=');
    }
    if let Ok(bytes) = general_purpose::STANDARD.decode(&padded) {
        return bytes;
    }

    // 4. Try URL Safe
    if let Ok(bytes) = general_purpose::URL_SAFE.decode(input) {
        return bytes;
    }
    if let Ok(bytes) = general_purpose::URL_SAFE_NO_PAD.decode(input) {
        return bytes;
    }

    // Fallback: return original bytes
    input.as_bytes().to_vec()
}
