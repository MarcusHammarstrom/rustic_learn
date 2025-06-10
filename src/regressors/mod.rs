use nalgebra::{DMatrix};

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