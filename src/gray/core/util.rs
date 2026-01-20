use anyhow::{ensure, Result};
use opencv::core::{self as cv, MatTraitConst};
use opencv::imgproc;
use opencv::prelude::*;
use std::f64::consts::PI;

pub(crate) const TOLERANCE: f64 = 0.0000001;

pub(crate) fn size_center(size: cv::Size) -> cv::Point2d {
    cv::Point2d::new(
        (size.width as f64 - 1.0) / 2.0,
        (size.height as f64 - 1.0) / 2.0,
    )
}

pub(crate) fn size_angle_step(size: cv::Size) -> f64 {
    let denom = size.width.max(size.height) as f64;
    (2.0 / denom).atan() * 180.0 / PI
}

pub(crate) fn get_rotation_matrix_2d(center: cv::Point2f, angle: f64) -> Result<cv::Mat> {
    let radians = angle * PI / 180.0;
    let alpha = radians.cos();
    let beta = radians.sin();

    let mut rotate = cv::Mat::zeros(2, 3, cv::CV_64F)?.to_mat()?;
    *rotate.at_2d_mut::<f64>(0, 0)? = alpha;
    *rotate.at_2d_mut::<f64>(0, 1)? = beta;
    *rotate.at_2d_mut::<f64>(0, 2)? = (1.0 - alpha) * center.x as f64 - beta * center.y as f64;
    *rotate.at_2d_mut::<f64>(1, 0)? = -beta;
    *rotate.at_2d_mut::<f64>(1, 1)? = alpha;
    *rotate.at_2d_mut::<f64>(1, 2)? = beta * center.x as f64 + (1.0 - alpha) * center.y as f64;

    Ok(rotate)
}

pub(crate) fn transform_point2d(point: cv::Point2d, rotate: &cv::Mat) -> Result<cv::Point2d> {
    let a00 = *rotate.at_2d::<f64>(0, 0)?;
    let a01 = *rotate.at_2d::<f64>(0, 1)?;
    let a02 = *rotate.at_2d::<f64>(0, 2)?;
    let a10 = *rotate.at_2d::<f64>(1, 0)?;
    let a11 = *rotate.at_2d::<f64>(1, 1)?;
    let a12 = *rotate.at_2d::<f64>(1, 2)?;

    let x = point.x * a00 + point.y * a01 + a02;
    let y = point.x * a10 + point.y * a11 + a12;
    Ok(cv::Point2d::new(x, y))
}

pub(crate) fn transform_with_center(
    point: cv::Point2d,
    center: cv::Point2d,
    angle: f64,
) -> Result<cv::Point2d> {
    let rotate = get_rotation_matrix_2d(cv::Point2f::new(center.x as f32, center.y as f32), angle)?;
    transform_point2d(point, &rotate)
}

pub(crate) fn compute_rotation_size(
    dst_size: cv::Size,
    template_size: cv::Size,
    mut angle: f64,
    rotate: &cv::Mat,
) -> Result<cv::Size> {
    if angle > 360.0 {
        angle -= 360.0;
    } else if angle < 0.0 {
        angle += 360.0;
    }

    if (angle.abs() - 90.0).abs() < TOLERANCE || (angle.abs() - 270.0).abs() < TOLERANCE {
        return Ok(cv::Size::new(dst_size.height, dst_size.width));
    }

    if angle.abs() < TOLERANCE || (angle.abs() - 180.0).abs() < TOLERANCE {
        return Ok(dst_size);
    }

    let points = [
        transform_point2d(cv::Point2d::new(0.0, 0.0), rotate)?,
        transform_point2d(cv::Point2d::new(dst_size.width as f64 - 1.0, 0.0), rotate)?,
        transform_point2d(cv::Point2d::new(0.0, dst_size.height as f64 - 1.0), rotate)?,
        transform_point2d(
            cv::Point2d::new(dst_size.width as f64 - 1.0, dst_size.height as f64 - 1.0),
            rotate,
        )?,
    ];

    let mut min = cv::Point2d::new(points[0].x, points[0].y);
    let mut max = cv::Point2d::new(points[0].x, points[0].y);
    for point in points.iter().skip(1) {
        min.x = min.x.min(point.x);
        min.y = min.y.min(point.y);
        max.x = max.x.max(point.x);
        max.y = max.y.max(point.y);
    }

    let center = size_center(dst_size);
    let width = (template_size.width as f64 - 1.0) / 2.0;
    let height = (template_size.height as f64 - 1.0) / 2.0;

    let half_height = (max.y - center.y - width).ceil() as i32;
    let half_width = (max.x - center.x - height).ceil() as i32;

    let mut size = cv::Size::new(half_width * 2, half_height * 2);
    let wrong_size = (template_size.width < size.width && template_size.height > size.height)
        || (template_size.width > size.width && template_size.height < size.height)
        || (template_size.width * template_size.height > size.width * size.height);
    if wrong_size {
        size = cv::Size::new(
            (max.x - min.x + 0.5).round() as i32,
            (max.y - min.y + 0.5).round() as i32,
        );
    }

    Ok(size)
}

pub(crate) fn crop_rotated_roi(
    src: &cv::Mat,
    template_size: cv::Size,
    top_left: cv::Point2d,
    rotate: &mut cv::Mat,
) -> Result<cv::Mat> {
    let point = transform_point2d(top_left, rotate)?;
    let padding_size = cv::Size::new(template_size.width + 6, template_size.height + 6);

    *rotate.at_2d_mut::<f64>(0, 2)? -= point.x - 3.0;
    *rotate.at_2d_mut::<f64>(1, 2)? -= point.y - 3.0;

    let mut roi = cv::Mat::default();
    imgproc::warp_affine_def(src, &mut roi, rotate, padding_size)?;
    Ok(roi)
}

pub(crate) fn compute_subpixel(score: &cv::Mat) -> Result<cv::Point2f> {
    let data = score.data_typed::<f32>()?;
    if data.len() < 9 {
        return Ok(cv::Point2f::new(0.0, 0.0));
    }

    let gx = (-data[0] + data[2] - data[3] + data[5] - data[6] + data[8]) / 3.0;
    let gy = (data[6] + data[7] + data[8] - data[0] - data[1] - data[2]) / 3.0;
    let gxx = (data[0] - 2.0 * data[1] + data[2] + data[3] - 2.0 * data[4] + data[5] + data[6]
        - 2.0 * data[7]
        + data[8])
        / 6.0;
    let gxy = (-data[0] + data[2] + data[6] - data[8]) / 2.0;
    let gyy = (data[0] + data[1] + data[2] - 2.0 * (data[3] + data[4] + data[5])
        + data[6]
        + data[7]
        + data[8])
        / 6.0;

    let trace = gxx + gyy;
    let disc = ((gxx - gyy) * (gxx - gyy) + 4.0 * gxy * gxy).sqrt();
    let lambda1 = (trace + disc) / 2.0;
    let lambda2 = (trace - disc) / 2.0;

    let (mut nx, mut ny) = if gxy.abs() > f32::EPSILON {
        if lambda1.abs() >= lambda2.abs() {
            (lambda1 - gyy, gxy)
        } else {
            (lambda2 - gyy, gxy)
        }
    } else if gxx.abs() >= gyy.abs() {
        (1.0, 0.0)
    } else {
        (0.0, 1.0)
    };

    let norm = (nx * nx + ny * ny).sqrt();
    if norm != 0.0 {
        nx /= norm;
        ny /= norm;
    }

    let denominator = gxx * nx * nx + 2.0 * gxy * nx * ny + gyy * ny * ny;
    if denominator == 0.0 {
        return Ok(cv::Point2f::new(0.0, 0.0));
    }

    let t = -(gx * nx + gy * ny) / denominator;
    Ok(cv::Point2f::new(t * nx, t * ny))
}

pub(crate) fn resize_template(template: &cv::Mat, scale: f64) -> Result<cv::Mat> {
    let width = (template.cols() as f64 * scale).round() as i32;
    let height = (template.rows() as f64 * scale).round() as i32;
    ensure!(width > 0 && height > 0, "scaled template size is invalid");

    let interpolation = if scale >= 1.0 {
        imgproc::INTER_LINEAR
    } else {
        imgproc::INTER_AREA
    };

    let mut resized = cv::Mat::default();
    imgproc::resize(
        template,
        &mut resized,
        cv::Size::new(width, height),
        0.0,
        0.0,
        interpolation,
    )?;
    Ok(resized)
}

pub(crate) fn next_max_loc_mat(
    score: &cv::Mat,
    pos: cv::Point,
    template_size: cv::Size,
    max_overlap: f64,
    max_score: &mut f64,
    max_pos: &mut cv::Point,
) -> Result<()> {
    let alone = 1.0 - max_overlap;
    let offset = cv::Point::new(
        (template_size.width as f64 * alone) as i32,
        (template_size.height as f64 * alone) as i32,
    );
    let size = cv::Size::new(
        (2.0 * template_size.width as f64 * alone) as i32,
        (2.0 * template_size.height as f64 * alone) as i32,
    );
    let rect_ignore = cv::Rect::new(pos.x - offset.x, pos.y - offset.y, size.width, size.height);

    let mut score = score.clone();
    imgproc::rectangle(
        &mut score,
        rect_ignore,
        cv::Scalar::all(-1.0),
        imgproc::FILLED,
        imgproc::LINE_8,
        0,
    )?;
    cv::min_max_loc(
        &score,
        None,
        Some(max_score),
        None,
        Some(max_pos),
        &cv::no_array(),
    )?;
    Ok(())
}

pub(crate) fn ccoeff_denominator(
    src: &cv::Mat,
    template_size: cv::Size,
    result: &mut cv::Mat,
    mean: f64,
    normal: f64,
    inv_area: f64,
    equal1: bool,
) -> Result<()> {
    if equal1 {
        result.set_to(&cv::Scalar::all(1.0), &cv::no_array())?;
        return Ok(());
    }

    let mut sum = cv::Mat::default();
    let mut sqsum = cv::Mat::default();
    imgproc::integral2(src, &mut sum, &mut sqsum, cv::CV_64F, cv::CV_64F)?;

    let sum_step = sum.step1_def()? as usize;
    let sqsum_step = sqsum.step1_def()? as usize;
    let result_step = result.step1_def()? as usize;

    let result_rows = result.rows() as usize;
    let result_cols = result.cols() as usize;
    let sum_data = sum.data_typed::<f64>()?;
    let sqsum_data = sqsum.data_typed::<f64>()?;
    let result_data = result.data_typed_mut::<f32>()?;

    let width = template_size.width as usize;
    let height = template_size.height as usize;
    let eps = f32::EPSILON as f64;

    for y in 0..result_rows {
        let sum_row = y * sum_step;
        let sum_row_bottom = (y + height) * sum_step;
        let sqsum_row = y * sqsum_step;
        let sqsum_row_bottom = (y + height) * sqsum_step;
        let result_row = y * result_step;
        for x in 0..result_cols {
            let idx_top = sum_row + x;
            let idx_bottom = sum_row_bottom + x;
            let part_sum = sum_data[idx_top] - sum_data[idx_top + width] - sum_data[idx_bottom]
                + sum_data[idx_bottom + width];

            let result_idx = result_row + x;
            let score = result_data[result_idx] as f64;
            let numerator = score - part_sum * mean;

            let part_sq_sum = sqsum_data[sqsum_row + x]
                - sqsum_data[sqsum_row + x + width]
                - sqsum_data[sqsum_row_bottom + x]
                + sqsum_data[sqsum_row_bottom + x + width];
            let part_sq_normal = part_sq_sum - part_sum * part_sum * inv_area;

            let diff = part_sq_normal.max(0.0);
            let denominator = if diff <= f64::min(0.5, 10.0 * eps * part_sq_sum) {
                0.0
            } else {
                diff.sqrt() * normal
            };

            if numerator.abs() < denominator {
                result_data[result_idx] = (numerator / denominator) as f32;
            } else if numerator.abs() < denominator * 1.125 {
                result_data[result_idx] = if numerator > 0.0 { 1.0 } else { -1.0 };
            } else {
                result_data[result_idx] = 0.0;
            }
        }
    }

    Ok(())
}
