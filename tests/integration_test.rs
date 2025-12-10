use breeze_asr_rs::BreezeASR;
use std::path::Path;

#[test]
fn test_audio_preprocessing() {
    // This test verifies that we can initialize the audio processor (which builds filterbanks)
    // without crashing. Actual audio loading requires a file.
    let _processor = breeze_asr_rs::audio::AudioProcessor::new();
    assert!(_processor.is_ok());
}

#[test]
fn test_tokenizer_loading() {
    // Requires the tokens.txt file. Since we don't have it locally in the repo (it's downloaded),
    // we can only test this if we mock it or if we assume network access (which we have).
    // However, unit tests shouldn't depend on large downloads usually.
    // We will just verify the struct exists and compiles.
    // For a real test, we would write a dummy tokens.txt
    
    use std::io::Write;
    let mut file = std::fs::File::create("test_tokens.txt").unwrap();
    writeln!(file, "hello").unwrap();
    writeln!(file, "world").unwrap();
    
    let tokenizer = breeze_asr_rs::tokenizer::Tokenizer::new("test_tokens.txt");
    assert!(tokenizer.is_ok());
    let tokenizer = tokenizer.unwrap();
    assert_eq!(tokenizer.decode(&[0, 1]), "helloworld");
    
    std::fs::remove_file("test_tokens.txt").unwrap();
}
