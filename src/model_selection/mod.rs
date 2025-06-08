mod fn train_test_split<T: Clone, U: Clone>(x: &Vec<Vec<T>>, y: &Vec<U>, test_ratio: Option<f32>, random_seed: Option<u64>) -> (Vec<Vec<T>>, Vec<U>, Vec<Vec<T>>, Vec<U>) {
    assert_eq!(x.len(), y.len(), "Features and labels must have the same length");
    
    let ratio = match test_ratio {
        Some(r) => {
            assert!(r > 0.0 && r < 1.0, "test_size must be between 0 and 1");
            r
        },
        None => 0.2, // Default to 20% if not provided
    };
    let mut rng = match random_seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(), // Use a random seed if not provided
    };
    let n_samples = x.len();
    let n_test = (n_samples as f32 * ratio).round() as usize;

    let mut indices = (0..n_samples).collect::<Vec<usize>>();

    indices.shuffle(&mut rng);
    let test_indices = &indices[..n_test];
    let train_indices = &indices[n_test..];

    let mut x_train = Vec::with_capacity(n_samples - n_test);
    let mut y_train = Vec::with_capacity(n_samples - n_test);
    let mut x_test = Vec::with_capacity(n_test);
    let mut y_test = Vec::with_capacity(n_test);


    for &i in test_indices {
        x_test.push(x[i].clone());
        y_test.push(y[i].clone());
    }
    for &i in train_indices {
        x_train.push(x[i].clone());
        y_train.push(y[i].clone());
    }

    (x_train, y_train, x_test, y_test)
}