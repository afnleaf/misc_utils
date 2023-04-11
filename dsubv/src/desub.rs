use std::fs::File;
use std::io::Read;
use std::{thread, time};
use term_size::dimensions;
use regex::Regex;

const FILENAME: &str = "heuristic.md";

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
    let sleep_time_dict = [(0, 0.18), (1, 0.5)];
    let mut sleep_time = 0.0;

    for word in document {
        if word.len() > max_word_length {
            max_word_length = word.len();
        }

        print!("\r{}{}", " ".repeat(max_word_length), "\r");
        if word != "\n" {
            print!("\r{}", word);
        }
        sleep_time = word.len() as f64 * 0.068;
        if sleep_time < 0.1 {
            sleep_time = 0.5;
        }
        thread::sleep(time::Duration::from_secs_f64(sleep_time));
    }
}

fn main() {
    let file_text = parse_file_to_clean_html(FILENAME);
    let document = turn_into_document(&file_text);
    let (width, _) = dimensions().unwrap_or((80, 24));
    print_out_document(document, width);
}
