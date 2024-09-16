use anyhow::Result;
use opencv as cv;
use opencv_match::prelude::{TryFromCv, TryIntoCv};

fn main() -> Result<()> {
    let img_tmp = image::open("./examples/up.png")?.to_rgba8();
    let img_src = image::open("./examples/sample.png")?.to_rgba8();

    let template = opencv_match::Template::new(
        opencv_match::convert::mat_to_grayscale(&img_tmp.clone().try_into_cv()?, true)?,
        Default::default(),
    )?;

    let start = std::time::Instant::now();
    let res = template.run_match(&opencv_match::convert::mat_to_grayscale(
        &img_src.try_into_cv()?,
        true,
    )?)?;
    println!("Elapsed: {:?}", start.elapsed());

    // We need to normalize the 0-1 values to 0-255.
    let mut normalized = cv::core::Mat::default();
    cv::core::normalize(
        &res,
        &mut normalized,
        0.0,
        255.0,
        cv::core::NORM_MINMAX,
        0,
        &cv::core::no_array(),
    )?;
    image::GrayImage::try_from_cv(normalized)?.save("./result.png")?;

    Ok(())
}
