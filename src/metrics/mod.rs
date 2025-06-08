pub mod distance_functions;

pub fn f1_score<T: Eq>(y_true: &[T], y_pred: &[T], labels: Vec<T>, pos_label: Option<usize>) -> f64 {
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut fn_ = 0.0;

    let label_pos = match pos_label {
        Some(label) => label,
        None => 0,
    };

    for (true_val, pred_val) in y_true.iter().zip(y_pred.iter()) {
        if true_val == pred_val {
            tp += 1.0;
        } 
        else if pred_val == &labels[label_pos] {
            fp += 1.0;
        } 
        else if true_val == &labels[label_pos] {
            fn_ += 1.0;
        }
    }

    let numberator = 2.0 * tp;
    let denominator = 2.0 * tp + fp + fn_;
    match denominator {
        0.0 => 0.0,
        _ => numberator / denominator,
    }
}
pub fn cosine_similarity(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    assert_eq!(a.len(), b.len(), "Vectors must be of the same length");
    let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0; // Avoid division by zero
    }
    
    dot_product / (norm_a * norm_b)
}