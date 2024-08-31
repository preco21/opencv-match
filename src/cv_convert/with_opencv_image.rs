use super::with_opencv::{MatExt, OpenCvElement};
use super::{TryFromCv, TryIntoCv};
use opencv::{core as cv_core, prelude::*};
use std::ops::Deref;

// ImageBuffer -> Mat
impl<P, Container> TryFromCv<image::ImageBuffer<P, Container>> for cv_core::Mat
where
    P: image::Pixel,
    P::Subpixel: OpenCvElement,
    Container: Deref<Target = [P::Subpixel]> + Clone,
{
    type Error = anyhow::Error;
    fn try_from_cv(from: image::ImageBuffer<P, Container>) -> Result<Self, Self::Error> {
        (&from).try_into_cv()
    }
}

// &ImageBuffer -> Mat
impl<P, Container> TryFromCv<&image::ImageBuffer<P, Container>> for cv_core::Mat
where
    P: image::Pixel,
    P::Subpixel: OpenCvElement,
    Container: Deref<Target = [P::Subpixel]> + Clone,
{
    type Error = anyhow::Error;
    fn try_from_cv(from: &image::ImageBuffer<P, Container>) -> Result<Self, Self::Error> {
        let (width, height) = from.dimensions();
        let cv_type = cv_core::CV_MAKETYPE(P::Subpixel::DEPTH, P::CHANNEL_COUNT as i32);
        let mat = unsafe {
            cv_core::Mat::new_rows_cols_with_data_unsafe(
                height as i32,
                width as i32,
                cv_type,
                from.as_ptr() as *mut _,
                cv_core::Mat_AUTO_STEP,
            )?
            .try_clone()?
        };
        Ok(mat)
    }
}

// &DynamicImage -> Mat
impl TryFromCv<&image::DynamicImage> for cv_core::Mat {
    type Error = anyhow::Error;

    fn try_from_cv(from: &image::DynamicImage) -> Result<Self, Self::Error> {
        use image::DynamicImage as D;

        let mat = match from {
            D::ImageLuma8(image) => image.try_into_cv()?,
            D::ImageLumaA8(image) => image.try_into_cv()?,
            D::ImageRgb8(image) => image.try_into_cv()?,
            D::ImageRgba8(image) => image.try_into_cv()?,
            D::ImageLuma16(image) => image.try_into_cv()?,
            D::ImageLumaA16(image) => image.try_into_cv()?,
            D::ImageRgb16(image) => image.try_into_cv()?,
            D::ImageRgba16(image) => image.try_into_cv()?,
            D::ImageRgb32F(image) => image.try_into_cv()?,
            D::ImageRgba32F(image) => image.try_into_cv()?,
            image => anyhow::bail!("the color type {:?} is not supported", image.color()),
        };
        Ok(mat)
    }
}

// DynamicImage -> Mat
impl TryFromCv<image::DynamicImage> for cv_core::Mat {
    type Error = anyhow::Error;
    fn try_from_cv(from: image::DynamicImage) -> Result<Self, Self::Error> {
        (&from).try_into_cv()
    }
}

// &Mat -> DynamicImage
impl TryFromCv<&cv_core::Mat> for image::DynamicImage {
    type Error = anyhow::Error;

    fn try_from_cv(from: &cv_core::Mat) -> Result<Self, Self::Error> {
        let rows = from.rows();
        let cols = from.cols();
        anyhow::ensure!(
            rows != -1 && cols != -1,
            "Mat with more than 2 dimensions is not supported."
        );

        let depth = from.depth();
        let n_channels = from.channels();
        let width = cols as u32;
        let height = rows as u32;

        let image: image::DynamicImage = match (depth, n_channels) {
            (cv_core::CV_8U, 1) => mat_to_image_buffer_gray::<u8>(from, width, height).into(),
            (cv_core::CV_16U, 1) => mat_to_image_buffer_gray::<u16>(from, width, height).into(),
            (cv_core::CV_8U, 3) => mat_to_image_buffer_rgb::<u8>(from, width, height).into(),
            (cv_core::CV_16U, 3) => mat_to_image_buffer_rgb::<u16>(from, width, height).into(),
            (cv_core::CV_32F, 3) => mat_to_image_buffer_rgb::<f32>(from, width, height).into(),
            _ => anyhow::bail!("Mat of type {} is not supported", from.type_name()),
        };

        Ok(image)
    }
}

// Mat -> DynamicImage
impl TryFromCv<cv_core::Mat> for image::DynamicImage {
    type Error = anyhow::Error;

    fn try_from_cv(from: cv_core::Mat) -> Result<Self, Self::Error> {
        (&from).try_into_cv()
    }
}

// &Mat -> gray ImageBuffer
impl<T> TryFromCv<&cv_core::Mat> for image::ImageBuffer<image::Luma<T>, Vec<T>>
where
    image::Luma<T>: image::Pixel,
    T: OpenCvElement + image::Primitive + DataType,
{
    type Error = anyhow::Error;

    fn try_from_cv(from: &cv_core::Mat) -> Result<Self, Self::Error> {
        let rows = from.rows();
        let cols = from.cols();
        anyhow::ensure!(
            rows != -1 && cols != -1,
            "Mat with more than 2 dimensions is not supported."
        );

        let depth = from.depth();
        let n_channels = from.channels();
        let width = cols as u32;
        let height = rows as u32;

        anyhow::ensure!(
            n_channels == 1,
            "Unable to convert a multi-channel Mat to a gray image"
        );
        anyhow::ensure!(depth == T::DEPTH, "Subpixel type is not supported");

        let image = mat_to_image_buffer_gray::<T>(from, width, height);
        Ok(image)
    }
}

// Mat -> gray ImageBuffer
impl<T> TryFromCv<cv_core::Mat> for image::ImageBuffer<image::Luma<T>, Vec<T>>
where
    image::Luma<T>: image::Pixel,
    T: OpenCvElement + image::Primitive + DataType,
{
    type Error = anyhow::Error;

    fn try_from_cv(from: cv_core::Mat) -> Result<Self, Self::Error> {
        (&from).try_into_cv()
    }
}

// &Mat -> rgb ImageBuffer
impl<T> TryFromCv<&cv_core::Mat> for image::ImageBuffer<image::Rgb<T>, Vec<T>>
where
    image::Rgb<T>: image::Pixel<Subpixel = T>,
    T: OpenCvElement + image::Primitive + DataType,
{
    type Error = anyhow::Error;

    fn try_from_cv(from: &cv_core::Mat) -> Result<Self, Self::Error> {
        let rows = from.rows();
        let cols = from.cols();
        anyhow::ensure!(
            rows != -1 && cols != -1,
            "Mat with more than 2 dimensions is not supported."
        );

        let depth = from.depth();
        let n_channels = from.channels();
        let width = cols as u32;
        let height = rows as u32;

        anyhow::ensure!(
            n_channels == 3,
            "Expect 3 channels, but get {n_channels} channels"
        );
        anyhow::ensure!(depth == T::DEPTH, "Subpixel type is not supported");

        let image = mat_to_image_buffer_rgb::<T>(from, width, height);
        Ok(image)
    }
}

// Mat -> rgb ImageBuffer
impl<T> TryFromCv<cv_core::Mat> for image::ImageBuffer<image::Rgb<T>, Vec<T>>
where
    image::Rgb<T>: image::Pixel<Subpixel = T>,
    T: OpenCvElement + image::Primitive + DataType,
{
    type Error = anyhow::Error;

    fn try_from_cv(from: cv_core::Mat) -> Result<Self, Self::Error> {
        (&from).try_into_cv()
    }
}

// &Mat -> rgba u8 ImageBuffer
impl TryFromCv<&cv_core::Mat> for image::ImageBuffer<image::Rgba<u8>, Vec<u8>> {
    type Error = anyhow::Error;

    fn try_from_cv(from: &cv_core::Mat) -> Result<Self, Self::Error> {
        let rows = from.rows();
        let cols = from.cols();
        anyhow::ensure!(
            rows != -1 && cols != -1,
            "Mat with more than 2 dimensions is not supported."
        );

        let depth = from.depth();
        let n_channels = from.channels();
        let width = cols as u32;
        let height = rows as u32;

        anyhow::ensure!(
            n_channels == 4,
            "Expect 4 channels, but get {n_channels} channels"
        );
        anyhow::ensure!(depth == u8::DEPTH, "Subpixel type is not supported");

        let image = mat_to_image_buffer_rgba_u8(from, width, height);
        Ok(image)
    }
}

// Mat -> rgba u8 ImageBuffer
impl TryFromCv<cv_core::Mat> for image::ImageBuffer<image::Rgba<u8>, Vec<u8>> {
    type Error = anyhow::Error;

    fn try_from_cv(from: cv_core::Mat) -> Result<Self, Self::Error> {
        (&from).try_into_cv()
    }
}

// Utility functions
fn mat_to_image_buffer_gray<T>(
    mat: &cv_core::Mat,
    width: u32,
    height: u32,
) -> image::ImageBuffer<image::Luma<T>, Vec<T>>
where
    T: image::Primitive + OpenCvElement + DataType,
{
    type Image<T> = image::ImageBuffer<image::Luma<T>, Vec<T>>;

    match mat.as_slice::<T>() {
        Ok(slice) => Image::<T>::from_vec(width, height, slice.to_vec()).unwrap(),
        Err(_) => Image::<T>::from_fn(width, height, |col, row| {
            let pixel: T = *mat.at_2d(row as i32, col as i32).unwrap();
            image::Luma([pixel])
        }),
    }
}

fn mat_to_image_buffer_rgb<T>(
    mat: &cv_core::Mat,
    width: u32,
    height: u32,
) -> image::ImageBuffer<image::Rgb<T>, Vec<T>>
where
    T: image::Primitive + OpenCvElement + DataType,
    image::Rgb<T>: image::Pixel<Subpixel = T>,
{
    type Image<T> = image::ImageBuffer<image::Rgb<T>, Vec<T>>;

    match mat.as_slice::<T>() {
        Ok(slice) => Image::<T>::from_vec(width, height, slice.to_vec()).unwrap(),
        Err(_) => Image::<T>::from_fn(width, height, |col, row| {
            let cv_core::Point3_::<T> { x, y, z } = *mat.at_2d(row as i32, col as i32).unwrap();
            image::Rgb([x, y, z])
        }),
    }
}

fn mat_to_image_buffer_rgba_u8(
    mat: &cv_core::Mat,
    width: u32,
    height: u32,
) -> image::ImageBuffer<image::Rgba<u8>, Vec<u8>> {
    image::ImageBuffer::<image::Rgba<u8>, Vec<u8>>::from_raw(
        width,
        height,
        mat.data_bytes().unwrap().to_owned(),
    )
    .unwrap()
}
