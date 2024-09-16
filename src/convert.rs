use crate::cv_convert::TryIntoCv;
use anyhow::Result;
use ndarray as nd;
use opencv::{self as cv, core::MatTraitConst};

pub fn mat_to_array2(mat: &cv::core::Mat) -> Result<nd::Array2<f32>> {
    let arr3: nd::Array3<f32> = mat.try_into_cv()?;
    let flatten = array3_to_array2(&arr3);
    Ok(flatten)
}

pub fn mat_to_grayscale(mat: &cv::core::Mat, rgba_color_space: bool) -> Result<cv::core::Mat> {
    let channels = mat.channels();
    if channels == 1 {
        // Mat is already grayscale, no need to convert.
        return Ok(mat.clone());
    }

    let mut res = cv::core::Mat::default();
    let code = match channels {
        3 => {
            if rgba_color_space {
                cv::imgproc::COLOR_RGB2GRAY
            } else {
                cv::imgproc::COLOR_BGR2GRAY
            }
        }
        4 => {
            if rgba_color_space {
                cv::imgproc::COLOR_RGBA2GRAY
            } else {
                cv::imgproc::COLOR_BGRA2GRAY
            }
        }
        _ => {
            return Err(anyhow::anyhow!("Unsupported number of channels."));
        }
    };

    cv::imgproc::cvt_color(&mat, &mut res, code, 0)?;
    Ok(res)
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
        .into_shape_with_order((height, (depth - 1) * width))
        .expect("failed to reshape array");
    array2.to_owned()
}
