use super::TryFromCv;
use half::f16;
use opencv::core::{self as cv, prelude::*, MatTraitConstManual};

pub use element_type::*;
mod element_type {
    use super::*;

    pub trait OpenCvElement {
        const DEPTH: i32;
    }

    impl OpenCvElement for u8 {
        const DEPTH: i32 = cv::CV_8U;
    }

    impl OpenCvElement for i8 {
        const DEPTH: i32 = cv::CV_8S;
    }

    impl OpenCvElement for u16 {
        const DEPTH: i32 = cv::CV_16U;
    }

    impl OpenCvElement for i16 {
        const DEPTH: i32 = cv::CV_16S;
    }

    impl OpenCvElement for i32 {
        const DEPTH: i32 = cv::CV_32S;
    }

    impl OpenCvElement for f16 {
        const DEPTH: i32 = cv::CV_16F;
    }

    impl OpenCvElement for f32 {
        const DEPTH: i32 = cv::CV_32F;
    }

    impl OpenCvElement for f64 {
        const DEPTH: i32 = cv::CV_64F;
    }
}

pub(crate) use mat_ext::*;
mod mat_ext {
    use anyhow::ensure;

    use super::*;

    pub trait MatExt {
        fn size_with_depth(&self) -> Vec<usize>;

        fn numel(&self) -> usize {
            self.size_with_depth().iter().product()
        }

        fn as_slice<T>(&self) -> anyhow::Result<&[T]>
        where
            T: OpenCvElement;

        fn type_name(&self) -> String;
    }

    impl MatExt for cv::Mat {
        fn size_with_depth(&self) -> Vec<usize> {
            let size = self.mat_size();
            let size = size.iter().map(|&dim| dim as usize);
            let channels = self.channels() as usize;
            size.chain([channels]).collect()
        }

        fn as_slice<T>(&self) -> anyhow::Result<&[T]>
        where
            T: OpenCvElement,
        {
            ensure!(self.depth() == T::DEPTH, "element type mismatch");
            ensure!(self.is_continuous(), "Mat data must be continuous");

            let numel = self.numel();
            let ptr = self.ptr(0)? as *const T;

            let slice = unsafe { std::slice::from_raw_parts(ptr, numel) };
            Ok(slice)
        }

        fn type_name(&self) -> String {
            cv::type_to_string(self.typ()).unwrap()
        }
    }
}

impl<T> TryFromCv<&cv::Mat> for cv::Point_<T>
where
    T: cv::DataType,
{
    type Error = anyhow::Error;

    fn try_from_cv(from: &cv::Mat) -> anyhow::Result<Self> {
        let slice = from.data_typed::<T>()?;
        anyhow::ensure!(slice.len() == 2, "invalid length");
        let point = Self {
            x: slice[0],
            y: slice[1],
        };
        Ok(point)
    }
}

impl<T> TryFromCv<cv::Mat> for cv::Point_<T>
where
    T: cv::DataType,
{
    type Error = anyhow::Error;

    fn try_from_cv(from: cv::Mat) -> anyhow::Result<Self> {
        TryFromCv::try_from_cv(&from)
    }
}

impl<T> TryFromCv<&cv::Mat> for cv::Point3_<T>
where
    T: cv::DataType,
{
    type Error = anyhow::Error;

    fn try_from_cv(from: &cv::Mat) -> anyhow::Result<Self> {
        let slice = from.data_typed::<T>()?;
        anyhow::ensure!(slice.len() == 3, "invalid length");
        let point = Self {
            x: slice[0],
            y: slice[1],
            z: slice[2],
        };
        Ok(point)
    }
}

impl<T> TryFromCv<cv::Mat> for cv::Point3_<T>
where
    T: cv::DataType,
{
    type Error = anyhow::Error;

    fn try_from_cv(from: cv::Mat) -> anyhow::Result<Self> {
        TryFromCv::try_from_cv(&from)
    }
}

impl<T> TryFromCv<&cv::Point_<T>> for cv::Mat
where
    T: cv::DataType,
{
    type Error = anyhow::Error;

    fn try_from_cv(from: &cv::Point_<T>) -> anyhow::Result<Self> {
        let cv::Point_ { x, y, .. } = *from;
        let point = [x, y];
        let mat = cv::Mat::from_slice(&point)?;
        Ok(mat.clone_pointee())
    }
}

impl<T> TryFromCv<cv::Point_<T>> for cv::Mat
where
    T: cv::DataType,
{
    type Error = anyhow::Error;

    fn try_from_cv(from: cv::Point_<T>) -> anyhow::Result<Self> {
        TryFromCv::try_from_cv(&from)
    }
}

impl<T> TryFromCv<&cv::Point3_<T>> for cv::Mat
where
    T: cv::DataType,
{
    type Error = anyhow::Error;

    fn try_from_cv(from: &cv::Point3_<T>) -> anyhow::Result<Self> {
        let cv::Point3_ { x, y, z, .. } = *from;
        let point = [x, y, z];
        let mat = cv::Mat::from_slice(&point)?;
        Ok(mat.clone_pointee())
    }
}

impl<T> TryFromCv<cv::Point3_<T>> for cv::Mat
where
    T: cv::DataType,
{
    type Error = anyhow::Error;

    fn try_from_cv(from: cv::Point3_<T>) -> anyhow::Result<Self> {
        TryFromCv::try_from_cv(&from)
    }
}
