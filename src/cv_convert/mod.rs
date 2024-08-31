mod traits;
pub use traits::*;

pub mod with_opencv;
pub mod with_opencv_image;
pub mod with_opencv_ndarray;

#[cfg(feature = "windows-capture")]
pub mod with_opencv_windows_capture;
