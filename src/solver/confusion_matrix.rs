//! TODO: DOC
use std::collections::VecDeque;
use std::fmt;

use co::SharedTensor;
use util::native_backend;
/// A [ConfusionMatrix][wiki].
///
/// [wiki]: https://en.wikipedia.org/wiki/Confusion_matrix
#[derive(Debug)]
pub struct ConfusionMatrix {
    num_classes: usize,

    /// maximum number of samples held
    capacity: Option<usize>,
    samples: VecDeque<Sample>,
}

impl ConfusionMatrix {
    /// Create a ConfusionMatrix that analyzes the prediction of `num_classes` classes.
    pub fn new(num_classes: usize) -> ConfusionMatrix {
        ConfusionMatrix {
            num_classes: num_classes,
            capacity: None,
            samples: VecDeque::new(),
        }
    }

    /// Add a sample by providing the expected `target` class and the `prediction`.
    pub fn add_sample(&mut self, prediction: usize, target: usize) {
        if self.capacity.is_some() && self.samples.len() >= self.capacity.unwrap() {
            self.samples.pop_front();
        }
        self.samples.push_back(Sample { prediction: prediction, target: target });
    }

    /// Add a batch of samples.
    ///
    /// See [add_sample](#method.add_sample).
    pub fn add_samples(&mut self, predictions: &[usize], targets: &[usize]) {
        for (&prediction, &target) in predictions.iter().zip(targets.iter()) {
            self.add_sample(prediction, target)
        }
    }

    /// Get the predicted classes from the output of a network.
    ///
    /// The prediction for each sample of the batch is found by
    /// determining which output value had the smallest loss.
    pub fn get_predictions(&self, network_out: &mut SharedTensor<f32>) -> Vec<usize> {
        let native_infered = network_out.read(native_backend().device()).unwrap()
            .as_native().unwrap();
        let predictions_slice = native_infered.as_slice::<f32>();

        let mut predictions = Vec::<usize>::new();
        for batch_predictions in predictions_slice.chunks(self.num_classes) {
            let mut enumerated_predictions = batch_predictions.iter().enumerate().collect::<Vec<_>>();
            enumerated_predictions.sort_by(|&(_, one), &(_, two)| one.partial_cmp(two).unwrap_or(::std::cmp::Ordering::Equal)); // find index of prediction
            predictions.push(enumerated_predictions.last().unwrap().0)
        }
        predictions
    }

    /// Set the `capacity` of the ConfusionMatrix
    pub fn set_capacity(&mut self, capacity: Option<usize>) {
        self.capacity = capacity;
        // TODO: truncate if over capacity
    }

    /// Return all collected samples.
    pub fn samples(&self) -> &VecDeque<Sample> {
        &self.samples
    }

    /// Return the accuracy of the collected predictions.
    pub fn accuracy(&self) -> Accuracy {
        let num_samples = self.samples.len();
        let num_correct = self.samples.iter().filter(|&&s| s.correct()).count();
        Accuracy { num_samples: num_samples, num_correct: num_correct }
    }
}

/// A single prediction Sample.
#[derive(Debug, Clone, Copy)]
pub struct Sample {
    prediction: usize,
    target: usize,
}

impl Sample {
    /// Returns if the prediction is equal to the expected target.
    pub fn correct(&self) -> bool {
        self.prediction == self.target
    }
}

impl fmt::Display for Sample {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Prediction: {:?}, Target: {:?}", self.prediction, self.target)
    }
}

#[derive(Debug, Clone, Copy)]
/// The accuracy of the predictions in a ConfusionMatrix.
///
/// Used to print the accuracy.
pub struct Accuracy {
    num_samples: usize,
    num_correct: usize,
}

impl Accuracy {
    fn ratio(&self) -> f32 {
        (self.num_correct as f32) / (self.num_samples as f32) * 100f32
    }
}

impl fmt::Display for Accuracy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}/{:?} = {:.2?}%", self.num_correct, self.num_samples, self.ratio())
    }
}
