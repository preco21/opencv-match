use anyhow::Result;
use opencv::imgproc;
use opencv_match::prelude::*;
use opencv_match::{
    convert,
    fast_gray::{MatchConfig, ModelPyramid, NmsConfig},
};

fn main() -> Result<()> {
    let img_tmp = image::open("./examples/test.png")?.to_rgba8();
    let img_src = image::open("./examples/source-resized.png")?.to_rgba8();

    let template_mat = convert::mat_to_grayscale(&img_tmp.try_into_cv()?, true)?;
    let target_mat = convert::mat_to_grayscale(&img_src.clone().try_into_cv()?, true)?;
    let scales = vec![1.0, 0.9, 0.8, 0.7, 0.6, 0.5];
    let start = std::time::Instant::now();
    let pyramid = ModelPyramid::train(template_mat.clone(), scales)?;
    let poses = pyramid.find_best_matches(
        &target_mat,
        MatchConfig {
            min_score: 0.8,
            max_count: 50,
            ..Default::default()
        },
        NmsConfig {
            iou_threshold: 0.1,
            score_threshold: 0.0,
        },
    )?;
    let duration = start.elapsed();
    println!("Matching took: {:?}", duration);

    println!("matches: {}", poses.len());
    for pose in &poses {
        println!(
            "x={}, y={}, angle={}, score={}, scale={}",
            pose.pose.x, pose.pose.y, pose.pose.angle, pose.pose.score, pose.scale
        );
    }

    let mut output_mat: opencv::core::Mat = img_src.try_into_cv()?;
    for pose in &poses {
        let rect_size = opencv::core::Size2f::new(pose.pose.width, pose.pose.height);
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
