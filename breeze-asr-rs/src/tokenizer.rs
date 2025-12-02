use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use anyhow::Result;

pub struct Tokenizer {
    id_to_token: HashMap<i64, String>,
}

impl Tokenizer {
    pub fn new(path: &str) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut id_to_token = HashMap::new();

        for (i, line) in reader.lines().enumerate() {
            let line = line?;
            id_to_token.insert(i as i64, line);
        }

        Ok(Self { id_to_token })
    }

    pub fn decode(&self, ids: &[i64]) -> String {
        let mut text = String::new();
        for &id in ids {
            if let Some(token) = self.id_to_token.get(&id) {
                if token.starts_with("<|") && token.ends_with("|>") {
                    continue;
                }

                let clean_token = token.replace("Ġ", " ");
                text.push_str(&clean_token);
            }
        }
        text
    }
}
