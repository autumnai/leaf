pub struct Blob<T> {
    data: Vec<T>,
    diff: Vec<T>,
    shape: Vec<isize>,
}

impl <T> Blob<T> {
    pub fn new() -> Blob<T> {
        let shape = vec![0];
        return Blob::of_shape(shape);
    }

    pub fn of_shape(new_shape: Vec<isize>) -> Blob<T> {
        let mut blob = Blob {
            data: vec![],
            diff: vec![],
            shape: vec![0],
        };
        blob.reshape(new_shape);

        return blob;
    }

    pub fn reshape(&mut self, new_shape: Vec<isize>) {
        let mut new_capacity = 1;

        for dimension in new_shape.iter() {
            // not sure if dimension is a fitting description
            new_capacity *= *dimension;
        }
        self.shape = new_shape;
        if new_capacity > self.data.capacity() as isize {
            self.data = Vec::with_capacity(new_capacity as usize);
            self.diff = Vec::with_capacity(new_capacity as usize);
        }
    }

    pub fn shape_string(&self) -> String {
        let mut string: String = "".to_owned();
        for dim in self.shape.clone() {
            string.push_str(&dim.to_string());
            string.push_str(" ");
        }
        string.push_str("(");
        string.push_str(&self.shape.len().to_string());
        string.push_str(")");

        string
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn cpu_data(&self) -> &Vec<T> {
        &self.data
    }

    pub fn mutable_cpu_data(&mut self) -> &mut Vec<T> {
        return &mut self.data;
    }

    pub fn cpu_diff(&self) -> &Vec<T> {
        &self.diff
    }

    pub fn mutable_cpu_diff(&mut self) -> &mut Vec<T> {
        return &mut self.diff;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_basic() {
        let blob: Blob<f32> = Blob::new();
        assert_eq!(blob.data.capacity(), 0);
    }

    #[test]
    fn construction_of_shape() {
        let shape = vec![2, 3, 2];
        let blob: Blob<f32> = Blob::of_shape(shape);
        assert_eq!(12, blob.data.capacity());
    }

    #[test]
    fn shape_string() {
        let shape = vec![2, 3, 2];
        let blob: Blob<f32> = Blob::of_shape(shape);
        assert_eq!("2 3 2 (3)", blob.shape_string());
    }
}
