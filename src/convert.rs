use crate::TryIntoCv;
use anyhow::Result;
use ndarray as nd;
use opencv as cv;

pub fn mat_to_array2(mat: &cv::core::Mat) -> Result<nd::Array2<f32>> {
    let arr3: nd::Array3<f32> = mat.try_into_cv()?;
    let flatten = array3_to_array2(&arr3);
    Ok(flatten)
}

pub fn array3_to_array2<T>(array3: &nd::Array3<T>) -> nd::Array2<T>
where
    T: Clone,
{
    // Get the dimensions of the input array.
    let (depth, height, width) = array3.dim();
    // Slice the array to merge the first element of the third dimension into the second dimension.
    let merged_array = array3.slice(nd::s![1.., .., ..]).reversed_axes();
    // Reshape the merged array to a 2-dimensional array.
    let array2 = merged_array
        .into_shape((height, (depth - 1) * width))
        .expect("failed to reshape array");
    array2.to_owned()
}
