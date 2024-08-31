use super::with_opencv::{MatExt as _, OpenCvElement};
use super::{TryFromCv, TryIntoCv};
use ndarray as nd;
use opencv::{core as cv_core, prelude::*};

impl<'a, A, D> TryFromCv<&'a cv_core::Mat> for nd::ArrayView<'a, A, D>
where
    A: OpenCvElement,
    D: nd::Dimension,
{
    type Error = anyhow::Error;

    fn try_from_cv(from: &'a cv_core::Mat) -> Result<Self, Self::Error> {
        let src_shape = from.size_with_depth();
        let array = nd::ArrayViewD::from_shape(src_shape, from.as_slice()?)?;
        let array = array.into_dimensionality()?;
        Ok(array)
    }
}

impl<A, D> TryFromCv<&cv_core::Mat> for nd::Array<A, D>
where
    A: OpenCvElement + Clone,
    D: nd::Dimension,
{
    type Error = anyhow::Error;

    fn try_from_cv(from: &cv_core::Mat) -> Result<Self, Self::Error> {
        let src_shape = from.size_with_depth();
        let array = nd::ArrayViewD::from_shape(src_shape, from.as_slice()?)?;
        let array = array.into_dimensionality()?;
        let array = array.into_owned();
        Ok(array)
    }
}

impl<A, D> TryFromCv<cv_core::Mat> for nd::Array<A, D>
where
    A: OpenCvElement + Clone,
    D: nd::Dimension,
{
    type Error = anyhow::Error;

    fn try_from_cv(from: cv_core::Mat) -> Result<Self, Self::Error> {
        (&from).try_into_cv()
    }
}

impl<A, S, D> TryFromCv<&nd::ArrayBase<S, D>> for cv_core::Mat
where
    A: cv_core::DataType,
    S: nd::RawData<Elem = A> + nd::Data,
    D: nd::Dimension,
{
    type Error = anyhow::Error;

    fn try_from_cv(from: &nd::ArrayBase<S, D>) -> anyhow::Result<Self> {
        let shape_with_channels: Vec<i32> = from.shape().iter().map(|&sz| sz as i32).collect();
        let (channels, shape) = match shape_with_channels.split_last() {
            Some(split) => split,
            None => {
                return Ok(Mat::default());
            }
        };
        let array = from.as_standard_layout();
        let slice = array.as_slice().unwrap();
        let mat = cv_core::Mat::from_slice(slice)?
            .reshape_nd(*channels, shape)?
            .clone_pointee();
        Ok(mat)
    }
}

impl<A, S, D> TryFromCv<nd::ArrayBase<S, D>> for cv_core::Mat
where
    A: cv_core::DataType,
    S: nd::RawData<Elem = A> + nd::Data,
    D: nd::Dimension,
{
    type Error = anyhow::Error;

    fn try_from_cv(from: nd::ArrayBase<S, D>) -> anyhow::Result<Self> {
        (&from).try_into_cv()
    }
}
