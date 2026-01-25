use anyhow::Result;
use opencv as cv;
use opencv::core::{self, MatTraitConst};
use opencv::imgproc;
use opencv_match::prelude::*;
use opencv_match::v2::{AdaptiveConfig, ScaleConfig, TemplateModel, TemplateModelConfig};

fn main() -> Result<()> {
    let template_path = "./examples/up.png";
    let source_path = "./examples/sample.png";

    let template_img = image::open(template_path)?;
    let source_img = image::open(source_path)?;

    let template = ensure_bgr(&template_img.try_into_cv()?)?;
    let source = ensure_bgr(&source_img.try_into_cv()?)?;

    let mut config = TemplateModelConfig::default();
    config.threshold = 0.9;
    config.scale = ScaleConfig::range(0.5, 2.0, 70);
    config.max_matches = Some(50);
    config.adaptive = Some(AdaptiveConfig::default());

    let model = TemplateModel::new(template, config)?;
    let matches = model.match_all(&source)?;

    println!("Found {} matches.", matches.len());
    for (idx, m) in matches.iter().take(5).enumerate() {
        println!(
            "#{idx}: score={:.4}, scale={:.3}, pos=({}, {}), size=({}, {})",
            m.score, m.scale, m.position.x, m.position.y, m.size.width, m.size.height
        );
    }

    let mut annotated = source.clone();
    for m in &matches {
        let top_left = m.position;
        imgproc::rectangle(
            &mut annotated,
            core::Rect::new(top_left.x, top_left.y, m.size.width, m.size.height),
            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            imgproc::LINE_8,
            0,
        )?;
    }

    let annotated = bgr_to_rgb(&annotated)?;
    let out_path = "./out-v2-matches.png";
    let out_img: image::DynamicImage = annotated.try_into_cv()?;
    out_img.save(out_path)?;
    println!("Wrote {out_path}");

    Ok(())
}

fn ensure_bgr(mat: &cv::core::Mat) -> Result<cv::core::Mat> {
    let channels = mat.channels();
    if channels == 3 {
        return Ok(mat.clone());
    }

    let mut converted = cv::core::Mat::default();
    let code = match channels {
        1 => imgproc::COLOR_GRAY2BGR,
        4 => imgproc::COLOR_RGBA2BGR,
        _ => anyhow::bail!("unsupported channel count: {channels}"),
    };
    imgproc::cvt_color(mat, &mut converted, code, 0)?;
    Ok(converted)
}

fn bgr_to_rgb(mat: &cv::core::Mat) -> Result<cv::core::Mat> {
    if mat.channels() != 3 {
        return Ok(mat.clone());
    }
    let mut converted = cv::core::Mat::default();
    imgproc::cvt_color(mat, &mut converted, imgproc::COLOR_BGR2RGB, 0)?;
    Ok(converted)
}
