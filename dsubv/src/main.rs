// Imports
use std::fs::File;
use std::io::Read;
use term_size::dimensions;
use regex::Regex;
use std::thread;
use std::time::Duration;
use std::io::{self, Write};


// &str type is string slice
const FILENAME: &str = "testfile.md";

// unwrap good
fn parse_file_to_clean_html(file_name: &str) -> String {
    let mut file = File::open(file_name).unwrap();
    let mut file_text = String::new();
    file.read_to_string(&mut file_text).unwrap();
    file_text
}


fn turn_into_document(file_text: &str) -> Vec<String> {
    let mut document = Vec::new();
    let lines = file_text.split('\n');
    let re = Regex::new(r"^#+\s").unwrap();

    for line in lines {
        if line != "" {
            if re.is_match(line) {
                document.push(line.to_owned());
                document.push("\n".to_owned());
            } else {
                let words = line.split_whitespace();
                for word in words {
                    document.push(word.to_owned());
                }
                document.push("\n".to_owned());
            }
        }
    }

    document
}


fn print_out_document(document: Vec<String>, max_word_size: usize) {
    let mut max_word_length = max_word_size;
    let mut sleep_time: u64;
    // to flush output
    let stdout = io::stdout();
    let mut handle = stdout.lock();

    for word in document {
        if word.len() > max_word_length {
            max_word_length = word.len();
        }

        print!("\r{}{}", " ".repeat(max_word_length), "\r");
        if word != "\n" {
            write!(handle, "\r{}", word).unwrap();
            handle.flush().unwrap();
        }
        
        sleep_time = word.len() as u64 * 68;
        if sleep_time < 500 {
            sleep_time = 500;
        }

        thread::sleep(Duration::from_millis(sleep_time));        
    }
}


fn main() {
    let file_text = parse_file_to_clean_html(FILENAME);
    let document = turn_into_document(&file_text);
    let (width, _) = dimensions().unwrap_or((80, 24));
    print_out_document(document, width);
}
