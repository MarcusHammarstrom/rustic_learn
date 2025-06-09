use std::fs::File;
use std::io::{BufRead, BufReader};

pub fn load_iris() -> (Vec<Vec<f64>>, Vec<String>) {
    let file_path = "./src/datasets/data/Iris.csv";
    parse_csv(file_path)
}
pub fn load_boston_housing() -> (Vec<Vec<f64>>, Vec<f64>) {
    let file_path = "./src/datasets/data/BostonHousing.csv";
    let (features, labels) = parse_csv(file_path);
    let new_labels: Vec<f64> = labels.iter()
        .map(|label| {
            label.parse::<f64>().unwrap_or_else(|_| {
                eprintln!("Warning: Could not parse label '{}', using 0.0", label);
                0.0
            })
        })
        .collect();
    (features, new_labels)
}

fn parse_csv(file_path: &str) -> (Vec<Vec<f64>>, Vec<String>) {
    let file = File::open(file_path).expect("Could not open file");
    let reader = BufReader::new(file);

    let mut features = Vec::new();
    let mut labels = Vec::new();
    for (line_num, line) in reader.lines().enumerate() {
        let line = line.expect("Could not read line");
        if !line.trim().is_empty() && line_num > 0 { 
            let (x, y) = parse_line(&line);
            features.push(x);
            labels.push(y);
        }
    }
    (features, labels)
}

fn parse_line(line: &str) -> (Vec<f64>, String) {
    let parts: Vec<&str> = line.split(',').collect();
    let mut x = Vec::new();
    let num_features = parts.len() - 1; // Last part is the label
    for i in 0..num_features {
        match parts[i].trim().parse::<f64>() {
            Ok(val) => x.push(val),
            Err(_) => {
                eprintln!("Warning: Could not parse feature {}", i+1);
                continue;
            }
        }
    }
    let y = parts[num_features].to_string();
    (x, y)
}