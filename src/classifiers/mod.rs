use super::metrics::distance_functions::euclidean_distance;

pub struct KnnClassifier<'a> {
    x_train: &'a Vec<Vec<f64>>,
    y_train: &'a Vec<String>,
    k: u32,
}
impl KnnClassifier<'_> {
    fn predict(&self, x_test: &Vec<Vec<f64>>) -> Vec<String> {
        let mut predictions = Vec::with_capacity(x_test.len());

        for test_sample in x_test {
            let mut distances: Vec<(f64, String)> = Vec::with_capacity(self.x_train.len());
            
            for (train_sample, label) in self.x_train.iter().zip(self.y_train.iter()) {
                let distance = euclidean_distance(&test_sample, train_sample);
                distances.push((distance, label.clone()));
            }

            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            
            let mut class_count = std::collections::HashMap::new();
            for i in 0..self.k as usize {
                let label = distances[i].1.clone();
                *class_count.entry(label).or_insert(0) += 1;
            }
            let mut max_count = 0;
            let mut predicted_label = distances[0].1.clone();
            for (label, count) in class_count.iter() {
                if *count > max_count {
                    max_count = *count;
                    predicted_label = label.clone();
                }
            }
            predictions.push(predicted_label);
        }

        predictions
    }
}