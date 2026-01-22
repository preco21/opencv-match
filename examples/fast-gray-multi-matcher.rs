use anyhow::Result;
use opencv::imgproc;
use opencv_match::prelude::*;
use opencv_match::{
    convert,
    fast_gray::{MatchConfig, ModelPyramid, NmsConfig},
};

fn main() -> Result<()> {
    let img_tmp = image::open("./examples/start.png")?.to_rgba8();
    let img_src = image::open("./examples/source2-resized.png")?.to_rgba8();

    let template_rgba: opencv::core::Mat = img_tmp.try_into_cv()?;
    let template_mat = convert::mat_to_grayscale(&template_rgba, true)?;
    let mut template_mask = opencv::core::Mat::default();
    opencv::core::extract_channel(&template_rgba, &mut template_mask, 3)?;
    let target_mat = convert::mat_to_grayscale(&img_src.clone().try_into_cv()?, true)?;
    let scales = vec![
        1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4,
    ];
    let start = std::time::Instant::now();

    let pyramid =
        ModelPyramid::train_with_mask(template_mat.clone(), template_mask.clone(), scales)?;
    let poses = pyramid.find_best_matches_adaptive(
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
        vec![1.0],
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
        let points = pose_points(&pose.pose);
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

fn pose_points(pose: &opencv_match::fast_gray::Pose) -> [opencv::core::Point2f; 4] {
    let angle = pose.angle as f64 * std::f64::consts::PI / 180.0;
    let cos = angle.cos();
    let sin = angle.sin();
    let x0 = pose.x as f64;
    let y0 = pose.y as f64;
    let w = pose.width as f64;
    let h = pose.height as f64;

    let points = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)];
    let mut out = [opencv::core::Point2f::new(0.0, 0.0); 4];
    for (idx, (dx, dy)) in points.iter().enumerate() {
        let x = x0 + dx * cos - dy * sin;
        let y = y0 + dx * sin + dy * cos;
        out[idx] = opencv::core::Point2f::new(x as f32, y as f32);
    }
    out
}
