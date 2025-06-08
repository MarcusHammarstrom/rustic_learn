pub fn euclidean_distance(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must be of the same length");
    let mut distance = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        distance += (x - y).powi(2);
    }
    distance.sqrt()
}
pub fn manhattan_distance(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must be of the same length");
    let mut distance = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        distance += (x-y).abs();
    }
    distance
}
pub fn chebyshev_distance(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
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