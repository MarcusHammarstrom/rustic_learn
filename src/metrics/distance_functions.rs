use super::cosine_similarity;

pub fn euclidean_distance(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    assert_eq!(a.len(), b.len(), "Vectors must be of the same length");
    let mut distance = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        distance += (x - y).powi(2);
    }
    distance.sqrt()
}
pub fn standardized_euclidean_distance(a: &Vec<f64>, b: &Vec<f64>, std_dev: &Vec<f64>) -> f64 {
    assert_eq!(a.len(), b.len(), "Vectors must be of the same length");
    assert_eq!(a.len(), std_dev.len(), "Vectors and standard deviations must be of the same length");
    let mut distance = 0.0;
    for ((&x, &y),  &s) in a.iter().zip(b.iter()).zip(std_dev.iter()) {
        if s == 0.0 {
            continue; // Avoid division by zero
        }
        distance += ((x - y) / s).powi(2);
    }
    distance.sqrt()
}
pub fn manhattan_distance(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    assert_eq!(a.len(), b.len(), "Vectors must be of the same length");
    let mut distance = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        distance += (x-y).abs();
    }
    distance
}
pub fn chebyshev_distance(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    assert_eq!(a.len(), b.len(), "Vectors must be of the same length");
    let mut max = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let value = (x-y).abs();
        if value > max {
            max = value;
        }
    }
    max
}
pub fn minkowski_distance(a: &Vec<f64>, b: &Vec<f64>, p: f64) -> f64 {
    assert_eq!(a.len(), b.len(), "Vectors must be of the same length");
    let mut sum = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        sum += (x - y).abs().powf(p);
    }
    sum.powf(1.0 / p)
}
pub fn canberra_distance(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    assert_eq!(a.len(), b.len(), "Vectors must be of the same length");
    let mut distance = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let abs_x_y = x.abs() + y.abs();
        if abs_x_y != 0.0 {
            distance += (x - y).abs() / abs_x_y;
        }
    }
    distance
}
pub fn cosine_distance(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    assert_eq!(a.len(), b.len(), "Vectors must be of the same length");
    1.0 - cosine_similarity(a, b)
}