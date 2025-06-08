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