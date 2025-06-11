use nalgebra::{DMatrix};
use super::metrics::distance_functions::euclidean_distance;

pub struct KnnRegression<'a> {
    x_train: &'a Vec<Vec<f64>>,
    y_train: &'a Vec<f64>,
    k: u32,
}
impl<'a> KnnRegression<'a> {
    pub fn new(x_train: &'a Vec<Vec<f64>>, y_train: &'a Vec<f64>, k: u32) -> KnnRegression<'a> {
        if k == 0 {
            panic!("k must be greater than 0");
        }
        if x_train.is_empty() || y_train.is_empty() {
            panic!("Training data cannot be empty");
        }
        if x_train.len() != y_train.len() {
            panic!("Number of training samples and labels must match");
        }
        KnnRegression {
            x_train,
            y_train,
            k,
        }
    }
    pub fn predict(&self, x_test: &Vec<Vec<f64>>) -> Vec<f64> {
        let mut predictions = Vec::with_capacity(x_test.len());

        for test_sample in x_test {
            let mut distances: Vec<(f64, f64)> = Vec::with_capacity(self.x_train.len());
            
            for (train_sample, &label) in self.x_train.iter().zip(self.y_train.iter()) {
                let distance = euclidean_distance(&test_sample, train_sample);
                distances.push((distance, label));
            }

            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            
            let mut sum = 0.0;
            for i in 0..self.k as usize {
                sum += distances[i].1;
            }
            let prediction = sum / self.k as f64;
            predictions.push(prediction);
        }
        predictions
    }
}
pub struct LinearRegression {
    coefficients: Option<Vec<f64>>,
}
impl LinearRegression {
    pub fn new() -> Self {
        LinearRegression { coefficients: None }
    }
    pub fn fit(&mut self, x: &Vec<Vec<f64>>, y: &Vec<f64>) {
        let flat_x: Vec<f64> = x.iter().flat_map(|v| v.iter()).cloned().collect();
        let rows = x.len();
        let cols = x[0].len();

        if rows == 0 || cols == 0 {
            panic!("Input data cannot be empty");
        }
        if rows != y.len() {
            panic!("Number of samples in x and y must match");
        }
        let matrix = DMatrix::from_vec(rows, cols, flat_x);
        let y_matrix = DMatrix::from_column_slice(rows, 1, y);
        let transpose = matrix.transpose();
        let coefficients = (&transpose * &matrix).try_inverse().unwrap() * &transpose * &y_matrix;
        self.coefficients = Some(coefficients.column(0).iter().map(|&x| x).collect());
    }
    pub fn predict(&self, x: &Vec<Vec<f64>>) -> Vec<f64> {
        if self.coefficients.is_none() {
            panic!("Model has not been fitted yet");
        }
        let coefficients = self.coefficients.as_ref().unwrap();
        let mut predictions = Vec::new();
        for row in x {
            if row.len() != coefficients.len() {
                panic!("Input feature length does not match model coefficients length");
            }
            let prediction: f64 = row.iter().zip(coefficients.iter()).map(|(a, b)| a * b).sum();
            predictions.push(prediction);
        }
        predictions
    }
}