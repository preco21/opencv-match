use anyhow::Result;
use opencv::core::MatTraitConst;
use opencv::imgproc;
use opencv_match::prelude::*;
use opencv_match::{
    convert,
    template_v2::{match_model_multi_scale, MatchConfig},
};

fn main() -> Result<()> {
    let img_tmp = image::open("./examples/test.png")?.to_rgba8();
    let img_src = image::open("./examples/source-resized.png")?.to_rgba8();

    let template_mat = convert::mat_to_grayscale(&img_tmp.try_into_cv()?, true)?;
    let target_mat = convert::mat_to_grayscale(&img_src.clone().try_into_cv()?, true)?;

    let max_scale = (target_mat.cols() as f64 / template_mat.cols() as f64)
        .min(target_mat.rows() as f64 / template_mat.rows() as f64);
    let scale_step = 0.25;
    let mut scales = Vec::new();
    let mut scale = 1.0;
    while scale <= max_scale && scale > 0.1 {
        scales.push(scale);
        scale -= scale_step;
    }

    let poses = match_model_multi_scale(
        &template_mat,
        &target_mat,
        &scales,
        MatchConfig {
            min_score: 0.9,
            max_count: 50,
            ..Default::default()
        },
    )?;

    println!("matches: {}", poses.len());
    for pose in &poses {
        println!(
            "x={}, y={}, angle={}, score={}, scale={}",
            pose.pose.x, pose.pose.y, pose.pose.angle, pose.pose.score, pose.scale
        );
    }

    let mut output_mat: opencv::core::Mat = img_src.try_into_cv()?;
    let template_size = template_mat.size()?;

    for pose in &poses {
        let rect_size = opencv::core::Size2f::new(
            (template_size.width as f64 * pose.scale) as f32,
            (template_size.height as f64 * pose.scale) as f32,
        );
        let rect = opencv::core::RotatedRect::new(
            opencv::core::Point2f::new(pose.pose.x, pose.pose.y),
            rect_size,
            -pose.pose.angle,
        )?;
        let mut points = [opencv::core::Point2f::new(0.0, 0.0); 4];
        rect.points(&mut points)?;
        for i in 0..4 {
            let p1 = points[i];
            let p2 = points[(i + 1) % 4];
            imgproc::line(
                &mut output_mat,
                opencv::core::Point::new(p1.x.round() as i32, p1.y.round() as i32),
                opencv::core::Point::new(p2.x.round() as i32, p2.y.round() as i32),
                opencv::core::Scalar::new(0.0, 255.0, 0.0, 255.0),
                2,
                imgproc::LINE_AA,
                0,
            )?;
        }
    }

    image::RgbaImage::try_from_cv(output_mat)?.save("out.png")?;

    Ok(())
}
