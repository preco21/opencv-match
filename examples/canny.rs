use anyhow::Result;
use image::{GrayImage, RgbImage, RgbaImage};
use opencv::{
    self as cv,
    imgproc::{COLOR_BGR2HSV, COLOR_RGB2HSV},
};
use opencv_match::{
    prelude::{TryFromCv, TryIntoCv},
    Template, TemplateConfig,
};

fn main() -> Result<()> {
    let template = image::open("./a.png")?.to_rgba8();
    let template_mat: cv::core::Mat = template.clone().try_into_cv()?;

    let mut template_edges_mat = cv::core::Mat::default();
    cv::imgproc::canny(&template_mat, &mut template_edges_mat, 0.0, 100.0, 3, true)?;

    let target = image::open("./hard.png")?.to_rgba8();
    let target_mat: cv::core::Mat = target.clone().try_into_cv()?;

    println!("target: {:?}", target_mat);
    println!("template: {:?}", template_mat);

    let mut hsv_template = cv::core::Mat::default();
    cv::imgproc::cvt_color(&target_mat, &mut hsv_template, COLOR_BGR2HSV, 0)?;

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
    cv::imgproc::canny(&target_mat, &mut target_edges_mat, 1.0, 100.0, 3, true)?;

    GrayImage::try_from_cv(&template_edges_mat)?.save("./result-canny-template.png")?;
    GrayImage::try_from_cv(&target_edges_mat)?.save("./result-canny-target.png")?;

    let matches = Template::new(
        template_edges_mat.clone(),
        TemplateConfig {
            threshold: 0.3,
            ..Default::default()
        },
    )?
    .find_best_matches(&target_edges_mat, Default::default())?;

    println!(
        "{:?}",
        matches
            .iter()
            .map(|m| format!(
                "x:{:?} y:{:?} (score: {:?})",
                m.position.x, m.position.y, m.score
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

    RgbImage::try_from_cv(&hsv_template)?.save("./result-hsv.png")?;

    Ok(())
}
