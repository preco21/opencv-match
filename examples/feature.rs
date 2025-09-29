use anyhow::Result;
use image::{GrayImage, RgbImage, RgbaImage};
use opencv::{
    self as cv,
    core::{
        KeyPointTraitConst, Mat, MatExprTraitConst, MatTraitConst, MatTraitManual, Point,
        VectorToVec,
    },
    imgproc::{COLOR_BGR2HSV, COLOR_RGB2HSV},
    prelude::{DescriptorMatcherTrait, Feature2DTrait},
};
use opencv_match::{
    prelude::{TryFromCv, TryIntoCv},
    Template, TemplateConfig,
};

fn main() -> Result<()> {
    let target = image::open("./examples/sample.png")?.to_rgba8();
    let target_mat: cv::core::Mat = target.clone().try_into_cv()?;
    let template = image::open("./examples/up.png")?.to_rgba8();
    let template_mat: cv::core::Mat = template.clone().try_into_cv()?;
    println!("target: {:?}", target_mat);
    println!("template: {:?}", template_mat);

    let mut dst_img: cv::core::Mat = target.try_into_cv()?;

    let matches = find_matching_boxes(
        &target_mat,
        &template_mat,
        DetectorMethod::Sift,
        Params {
            max_matching_objects: 10,
            sift_distance_threshold: 0.5,
            best_matches_points: 20,
        },
    )?;

    // for m in matches {
    //     cv::imgproc::rectangle(
    //         &mut dst_img,
    //         cv::core::Rect::from_point_size(m.position, m.dimension),
    //         cv::core::VecN([255., 255., 0., 0.]),
    //         2,
    //         cv::imgproc::LINE_8,
    //         0,
    //     )?;
    // }

    RgbaImage::try_from_cv(&dst_img)?.save("./result.png")?;

    Ok(())
}

#[derive(Clone, Copy)]
pub enum DetectorMethod {
    Sift,
    Orb,
}

#[derive(Clone, Copy)]
pub struct Params {
    pub max_matching_objects: i32,
    pub sift_distance_threshold: f32, // ratio test multiplier
    pub best_matches_points: usize,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            max_matching_objects: 10,
            sift_distance_threshold: 0.5,
            best_matches_points: 20,
        }
    }
}

/// Returns a Vec of 4-point quadrilaterals (each as a Mat of shape (4,1, CV_32FC2))
pub fn find_matching_boxes(
    target: &cv::core::Mat,
    template: &cv::core::Mat,
    detector_method: DetectorMethod,
    params: Params,
) -> Result<Vec<cv::core::Mat>> {
    // Prepare detector + matcher
    let (mut sift, mut orb, mut matcher) = match detector_method {
        DetectorMethod::Sift => {
            // SIFT defaults; tweak if you need
            let sift = cv::features2d::SIFT::create(0, 3, 0.04, 10.0, 1.6, false)?;
            let matcher = cv::features2d::BFMatcher::new(cv::core::NORM_L2, false)?; // ratio test; no cross-check
            (Some(sift), None, matcher)
        }
        DetectorMethod::Orb => {
            // edgeThreshold/fastThreshold similar to your Python
            let mut orb = cv::features2d::ORB::create(
                500,
                1.2,
                8,
                10,
                0,
                2,
                cv::features2d::ORB_ScoreType::HARRIS_SCORE,
                31,
                5,
            )?;
            // (fastThreshold is embedded in ORB params; OpenCV’s ORB::setFastThreshold is not exposed, so tune create)
            let matcher = cv::features2d::BFMatcher::new(cv::core::NORM_HAMMING, true)?; // crossCheck = true
            (None, Some(orb), matcher)
        }
    };

    // Detect on template once
    let mut template_kps = cv::core::Vector::<cv::core::KeyPoint>::new();
    let mut template_desc = cv::core::Mat::default(); // query
    match detector_method {
        DetectorMethod::Sift => {
            sift.as_mut().unwrap().detect_and_compute(
                template,
                &opencv::core::no_array(),
                &mut template_kps,
                &mut template_desc,
                false,
            )?;
        }
        DetectorMethod::Orb => {
            orb.as_mut().unwrap().detect_and_compute(
                template,
                &opencv::core::no_array(),
                &mut template_kps,
                &mut template_desc,
                false,
            )?;
        }
    }

    let mut matched_boxes: Vec<opencv::core::Mat> = Vec::new();
    let mut matching_img = target.clone();

    for _ in 0..params.max_matching_objects {
        // Detect on current image
        let mut target_kps = opencv::core::Vector::<opencv::core::KeyPoint>::new();
        let mut target_desc = opencv::core::Mat::default();
        match detector_method {
            DetectorMethod::Sift => {
                sift.as_mut().unwrap().detect_and_compute(
                    &matching_img,
                    &opencv::core::no_array(),
                    &mut target_kps,
                    &mut target_desc,
                    false,
                )?;
            }
            DetectorMethod::Orb => {
                orb.as_mut().unwrap().detect_and_compute(
                    &matching_img,
                    &opencv::core::no_array(),
                    &mut target_kps,
                    &mut target_desc,
                    false,
                )?;
            }
        }

        // If descriptors are empty, we’re done
        if target_desc.empty() || template_desc.empty() {
            break;
        }

        // Match descriptors
        let mut good = Vec::new();
        match detector_method {
            DetectorMethod::Sift => {
                // KNN + ratio test
                let mut knn = cv::core::Vector::new();
                matcher.add(&target_desc)?;
                matcher.knn_match(
                    &template_desc,
                    &mut knn,
                    2,
                    &opencv::core::no_array(),
                    false,
                )?;
                for pair in knn {
                    if pair.len() == 2 {
                        let m = pair.get(0).unwrap();
                        let n = pair.get(1).unwrap();
                        if m.distance < params.sift_distance_threshold * n.distance {
                            good.push(m);
                        }
                    }
                }
                // Sort by distance and keep best N
                good.sort_by(|a, b| {
                    a.distance
                        .partial_cmp(&b.distance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                if good.len() > params.best_matches_points {
                    good.truncate(params.best_matches_points);
                }
            }
            DetectorMethod::Orb => {
                let mut matches = cv::core::Vector::new();
                matcher.add(&target_desc)?;
                matcher.match_(&template_desc, &mut matches, &opencv::core::no_array())?;
                let mut v = matches.to_vec();
                v.sort_by(|a, b| {
                    a.distance
                        .partial_cmp(&b.distance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                if v.len() > params.best_matches_points {
                    v.truncate(params.best_matches_points);
                }
                good = v;
            }
        }

        if good.len() < 4 {
            // Not enough correspondences for homography
            break;
        }

        // Build point sets
        let mut pts1 = opencv::core::Vector::<opencv::core::Point2f>::new();
        let mut pts2 = opencv::core::Vector::<opencv::core::Point2f>::new();
        for m in &good {
            let p1 = target_kps.get(m.query_idx as usize)?.pt();
            let p2 = template_kps.get(m.train_idx as usize)?.pt();
            pts1.push(opencv::core::Point2f::new(p1.x, p1.y));
            pts2.push(opencv::core::Point2f::new(p2.x, p2.y));
        }

        // Find homography (template -> image)
        let mut mask = cv::core::Mat::default();
        let h = match cv::calib3d::find_homography_ext(
            &pts2,
            &pts1,
            cv::calib3d::RANSAC,
            2.0,
            &mut mask,
            2000,
            0.995,
        ) {
            Ok(h) => h,
            Err(_) => break,
        };

        if h.empty() {
            break;
        }

        // Transform template corners
        let h_t = template.rows();
        let w_t = template.cols();
        let mut corners = Mat::default();
        {
            // shape: (4,1,CV_32FC2)
            corners = opencv::core::Mat::zeros(4, 1, cv::core::CV_32FC2)?.to_mat()?;
            let mut corner_data = corners.at_row_mut::<cv::core::Vec2f>(0)?;
            // (0,0)
            corner_data[0] = cv::core::Vec2f::from([0.0, 0.0]);
            // (0,h-1)
            corner_data[1] = cv::core::Vec2f::from([0.0, (h_t - 1) as f32]);
            // (w-1,h-1)
            corner_data[2] = cv::core::Vec2f::from([(w_t - 1) as f32, (h_t - 1) as f32]);
            // (w-1,0)
            corner_data[3] = cv::core::Vec2f::from([(w_t - 1) as f32, 0.0]);
        }
        let mut transformed = Mat::default();
        cv::core::perspective_transform(&corners, &mut transformed, &h)?;

        matched_boxes.push(transformed.clone());

        // ---- Mask out the found region and inpaint for next iteration ----
        // 1) Prepare a full-255 mask
        let mut gray = Mat::default();
        cv::imgproc::cvt_color(&matching_img, &mut gray, cv::imgproc::COLOR_BGR2GRAY, 0)?;
        let mut mask_u8 = Mat::new_rows_cols_with_default(
            gray.rows(),
            gray.cols(),
            cv::core::CV_8UC1,
            cv::core::Scalar::all(255.0),
        )?;

        // 2) Fill polygon (transformed corners) with 0
        // Convert transformed (float) to integer points
        let mut poly_pts = cv::core::Vector::<Point>::new();
        {
            let t = transformed.reshape(2, 4)?; // 4x2
            for i in 0..4 {
                let x = *t.at_2d::<f32>(i, 0)? as i32;
                let y = *t.at_2d::<f32>(i, 1)? as i32;
                poly_pts.push(Point::new(x, y));
            }
        }
        // let mut polys = cv::core::Vector::new();
        // polys.push(poly_pts);
        cv::imgproc::fill_poly(
            &mut mask_u8,
            &poly_pts,
            opencv::core::Scalar::all(0.0),
            cv::imgproc::LINE_AA,
            0,
            cv::core::Point::new(0, 0),
        )?;

        // 3) Invert to get inpaint mask = region to repair
        let mut inpaint_mask = Mat::default();
        cv::core::bitwise_not(&mask_u8, &mut inpaint_mask, &opencv::core::no_array())?;

        // 4) Inpaint on BGR image to erase found instance
        let mut repaired = Mat::default();
        cv::photo::inpaint(
            &matching_img,
            &inpaint_mask,
            &mut repaired,
            3.0,
            cv::photo::INPAINT_TELEA,
        )?;
        matching_img = repaired;
    }

    Ok(matched_boxes)
}
