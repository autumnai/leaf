pub struct Blob<T> {
    data: Vec<T>,
    shape: Vec<i32>
}

impl <T> Blob<T> {
    pub fn new() -> Blob<T> {
        let shape = vec![0];
        return Blob::of_shape(shape);
    }

    pub fn of_shape(new_shape: Vec<i32>) -> Blob<T> {
        let mut blob = Blob {
            data: vec![],
            shape: vec![0]
        };
        blob.reshape(new_shape);

        return blob;
    }

    pub fn reshape(&mut self, new_shape: Vec<i32>) {
        let mut new_capacity = 1;

        for dimension in new_shape.iter() { // not sure if dimension is a fitting description
            new_capacity *= *dimension;
        }
        self.shape = new_shape;
        if new_capacity > self.data.capacity() as i32 {
            self.data = Vec::with_capacity(new_capacity as usize);
        }
    }

    pub fn cpu_data(&self) -> &Vec<T> {
        &self.data
    }

    pub fn mutable_cpu_data(&mut self) -> &mut Vec<T> {
        return &mut self.data;
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
}
