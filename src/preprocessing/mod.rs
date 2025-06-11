pub fn standard_scale(data: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut scaled_data = data.clone();
    for i in 0..scaled_data[0].len() {
        let column: Vec<f64> = scaled_data.iter().map(|row| row[i]).collect();
        let mean: f64 = column.iter().sum::<f64>() / column.len() as f64;
        let std_dev: f64 = (column.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / column.len() as f64).sqrt();
        for row in &mut scaled_data {
            row[i] = (row[i] - mean) / std_dev;
        }
    }
    scaled_data
}
pub fn min_max_scale(data: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut scaled_data = data.clone();
    for i in 0..scaled_data[0].len() {
        let column: Vec<f64> = scaled_data.iter().map(|row| row[i]).collect();
        let min: f64 = *column.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let max: f64 = *column.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        for row in &mut scaled_data {
            row[i] = (row[i] - min) / (max - min);
        }
    }
    scaled_data
}