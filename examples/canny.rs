use anyhow::Result;
use image::{GrayImage, RgbaImage};
use opencv::{self as cv};
use opencv_match::{
    prelude::{TryFromCv, TryIntoCv},
    FindBestMatchesConfig, Template, TemplateConfig,
};

fn main() -> Result<()> {
    let template = image::open("./examples/up.png")?.to_rgba8();
    let template_mat: cv::core::Mat = template.clone().try_into_cv()?;

    let mut template_edges_mat = cv::core::Mat::default();
    cv::imgproc::canny(
        &template_mat,
        &mut template_edges_mat,
        100.0,
        200.0,
        3,
        true,
    )?;

    let target = image::open("./examples/sample.png")?.to_rgba8();
    let target_mat: cv::core::Mat = target.clone().try_into_cv()?;

    // Some research suggests that blurring the input image can help with edge detection.
    let mut target_input_blur_mat = cv::core::Mat::default();
    cv::imgproc::gaussian_blur(
        &target_mat,
        &mut target_input_blur_mat,
        cv::core::Size::new(1, 1),
        0.0,
        0.0,
        0,
    )?;

    let mut target_edges_mat = cv::core::Mat::default();
    cv::imgproc::canny(
        &target_input_blur_mat,
        &mut target_edges_mat,
        150.0,
        300.0,
        3,
        true,
    )?;

    GrayImage::try_from_cv(&template_edges_mat)?.save("./result-canny-template.png")?;
    GrayImage::try_from_cv(&target_edges_mat)?.save("./result-canny-target.png")?;

    let matches = Template::new(
        template_mat,
        TemplateConfig {
            threshold: 0.5,
            ..Default::default()
        },
    )?
    .find_best_matches(
        &target_mat,
        FindBestMatchesConfig {
            iou_threshold: 0.0,
            score_threshold: 0.0,
        },
    )?;

    println!(
        "{:?}",
        matches
            .iter()
            .map(|m| format!(
                "{:?} {:?} {:?} {:?}",
                m.position.x,
                m.position.y,
                m.position.x + m.dimension.width,
                m.position.y + m.dimension.height
            ))
            .collect::<Vec<_>>()
    );

    let mut dst_img: cv::core::Mat = target.try_into_cv()?;
    for m in matches {
        cv::imgproc::rectangle(
            &mut dst_img,
            cv::core::Rect::from_point_size(m.position, m.dimension),
            cv::core::VecN([255., 255., 0., 0.]),
            2,
            cv::imgproc::LINE_8,
            0,
        )?;
    }

    RgbaImage::try_from_cv(&dst_img)?.save("./result-canny.png")?;

    Ok(())
}
