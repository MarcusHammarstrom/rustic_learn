use rustic_learn::classifiers::KnnClassifier;
use rustic_learn::regressors::LinearRegression;
use rustic_learn::datasets::{load_boston_housing, load_iris};
use rustic_learn::model_selection::train_test_split;

fn main() {
    let (x, y) = load_iris();
    let (x_train, y_train, x_test, y_test) = train_test_split(&x, &y, Some(0.2), None);
    let knn = KnnClassifier::new(&x_train, &y_train, 3);
    let predictions = knn.predict(&x_test);

    let mut correct = 0.0;
    let total = predictions.len();
    for (predicted, actual) in predictions.iter().zip(y_test.iter()) {
        if predicted == actual {
            correct += 1.0;
        }
    }
    let accuracy = correct / total as f64;
    println!("Accuracy: {:.2}%", accuracy * 100.0);
    let (x, y) = load_boston_housing();
    let (x_train, y_train, x_test, y_test) = train_test_split(&x, &y, Some(0.2), None);
    let mut regressor = LinearRegression::new();
    regressor.fit(&x_train, &y_train);
    let predictions = regressor.predict(&x_test);
    let mut rmse = 0.0;
    for (predicted, actual) in predictions.iter().zip(y_test.iter()) {
        rmse += (predicted - actual).powi(2);
    }
    rmse = rmse / predictions.len() as f64;
    rmse = rmse.sqrt();
    println!("Root Mean Squared Error: {:.2}", rmse);
    println!("Prediction 1: {}, Actual 1: {}", predictions[0], y_test[0]);
    println!("Prediction 2: {}, Actual 1: {}", predictions[1], y_test[1]);   
}