use rustic_learn::metrics::distance_functions::euclidean_distance;

fn main() {
    let vec1 = vec![1.0, 2.0, 3.0];
    let vec2 = vec![4.0, 5.0, 6.0];
    
    let distance = euclidean_distance(&vec1, &vec2);
    println!("Euclidean Distance: {}", distance);
}