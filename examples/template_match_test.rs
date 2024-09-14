use anyhow::Result;
use opencv_match::prelude::*;

fn main() -> Result<()> {
    let img_tmp = image::open("./examples/up.png")?.to_rgba8();
    let img_src = image::open("./examples/sample.png")?.to_rgba8();

    let template = opencv_match::Template::new(img_tmp.try_into_cv()?, 0.8);

    let start = std::time::Instant::now();
    let res = template.run_match(&img_src.try_into_cv()?)?;
    println!("Elapsed: {:?}", start.elapsed());

    image::GrayImage::try_from_cv(&res)?.save("./result.png")?;

    Ok(())
}
