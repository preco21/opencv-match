use anyhow::Result;
use opencv_match::prelude::*;
use opencv_match::{
    convert, FindBestMatchesConfig, MultiMatcher, MultiMatcherDescriptor, Template, TemplateConfig,
    TemplatePyramid,
};

fn main() -> Result<()> {
    let img_tmp = image::open("./examples/up.png")?.to_rgba8();
    let img_src = image::open("./examples/sample.png")?.to_rgba8();

    let template_mat = convert::mat_to_grayscale(&img_tmp.try_into_cv()?, true)?;
    let target_mat = convert::mat_to_grayscale(&img_src.try_into_cv()?, true)?;

    let template = Template::new(
        template_mat,
        TemplateConfig {
            threshold: 0.85,
            matching_method: None,
        },
    )?;

    let pyramid = template.pyramid(vec![0.75, 1.0, 1.25])?;

    let descriptors = vec![
        MultiMatcherDescriptor::new("single".to_string(), template),
        MultiMatcherDescriptor::with_pyramid("pyramid".to_string(), pyramid),
    ];
    let matcher = MultiMatcher::new(descriptors)?;

    let matches = matcher.find_best_matches(
        &target_mat,
        FindBestMatchesConfig {
            iou_threshold: 0.1,
            score_threshold: 0.0,
        },
    )?;

    println!("matches: {}", matches.len());
    for m in matches {
        println!(
            "label={} scale={:?} pos=({}, {}) size=({}x{}) score={}",
            m.label,
            m.scale,
            m.result.position.x,
            m.result.position.y,
            m.result.dimension.width,
            m.result.dimension.height,
            m.result.score
        );
    }

    Ok(())
}
