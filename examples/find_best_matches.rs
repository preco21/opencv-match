use anyhow::Result;
use opencv_match::prelude::*;

fn main() -> Result<()> {
    let img_tmp = image::open("./examples/up.png")?.to_rgba8();
    let img_src = image::open("./examples/sample.png")?.to_rgba8();

    let template = opencv_match::Template::new(
        opencv_match::convert::mat_to_grayscale(&img_tmp.clone().try_into_cv()?, true)?,
        Default::default(),
    )?;

    let start = std::time::Instant::now();
    let res = template.find_best_matches(&opencv_match::convert::mat_to_grayscale(
        &img_src.try_into_cv()?,
        true,
    )?)?;
    println!("Elapsed: {:?}", start.elapsed());

    println!("Best matches: {:?}", res);

    Ok(())
}
