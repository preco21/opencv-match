use anyhow::{bail, ensure, Result};
use opencv::core::{self as cv, MatTraitConst};
use opencv::imgproc;
use opencv::prelude::*;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::f64::consts::PI;

const MIN_AREA: i32 = 128;
const TOLERANCE: f64 = 0.0000001;
const CANDIDATE: usize = 5;
const INVALID: f64 = -1.0;

#[derive(Debug, Clone, Copy)]
pub struct Pose {
    pub x: f32,
    pub y: f32,
    pub angle: f32,
    pub score: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct ScaledPose {
    pub pose: Pose,
    pub scale: f64,
}

#[derive(Debug, Clone)]
pub struct MatchConfig {
    /// Match start level (-1 for auto).
    pub level: i32,
    /// Rotation start angle.
    pub start_angle: f64,
    /// Rotation span angle.
    pub span_angle: f64,
    /// Overlap threshold.
    pub max_overlap: f64,
    /// Minimum match score.
    pub min_score: f64,
    /// Maximum number of candidates per angle.
    pub max_count: usize,
    /// Compute subpixel offset on the last level.
    pub subpixel: bool,
}

impl Default for MatchConfig {
    fn default() -> Self {
        Self {
            level: -1,
            start_angle: 0.0,
            span_angle: 360.0,
            max_overlap: 0.0,
            min_score: 0.7,
            max_count: 50,
            subpixel: true,
        }
    }
}

#[derive(Debug)]
pub struct GrayMatchModel {
    pyramids: Vec<cv::Mat>,
    mean: Vec<cv::Scalar>,
    normal: Vec<f64>,
    inv_area: Vec<f64>,
    equal1: Vec<bool>,
    border_color: u8,
}

impl GrayMatchModel {
    pub fn train(template: &cv::Mat, level: i32) -> Result<Self> {
        ensure!(!template.empty(), "template image is empty");
        ensure!(template.channels() == 1, "template image must be grayscale");

        let level = if level <= 0 {
            compute_layers(template.cols(), template.rows(), MIN_AREA)
        } else {
            level
        };

        if level < 1 {
            bail!("template image is too small for pyramid training");
        }

        let scale = 1_i32.checked_shl((level - 1) as u32).unwrap_or_default();
        let top_area = template.rows() * template.cols() / (scale * scale);
        ensure!(
            top_area >= MIN_AREA,
            "template image is too small for top pyramid level"
        );

        let mut pyramids = cv::Vector::<cv::Mat>::new();
        imgproc::build_pyramid_def(template, &mut pyramids, level)?;

        let mut model = GrayMatchModel {
            pyramids: Vec::with_capacity(pyramids.len()),
            mean: Vec::with_capacity(pyramids.len()),
            normal: Vec::with_capacity(pyramids.len()),
            inv_area: Vec::with_capacity(pyramids.len()),
            equal1: Vec::with_capacity(pyramids.len()),
            border_color: 0,
        };

        let mean_value = cv::mean(template, &cv::no_array())?.0[0];
        model.border_color = if mean_value < 128.0 { 255 } else { 0 };

        for i in 0..pyramids.len() {
            let pyramid = pyramids.get(i)?;

            let area = (pyramid.rows() * pyramid.cols()) as f64;
            let inv_area: f64 = 1.0 / area;

            let mut mean = cv::Scalar::default();
            let mut stddev = cv::Scalar::default();
            cv::mean_std_dev(&pyramid, &mut mean, &mut stddev, &cv::no_array())?;

            let std_normal = stddev.0[0] * stddev.0[0]
                + stddev.0[1] * stddev.0[1]
                + stddev.0[2] * stddev.0[2]
                + stddev.0[3] * stddev.0[3];
            let equal1 = std_normal < f64::EPSILON;
            let normal = std_normal.sqrt() / inv_area.sqrt();

            model.pyramids.push(pyramid);
            model.mean.push(mean);
            model.normal.push(normal);
            model.inv_area.push(inv_area);
            model.equal1.push(equal1);
        }

        Ok(model)
    }

    pub fn pyramid_levels(&self) -> usize {
        self.pyramids.len()
    }

    pub fn pyramid(&self, level: usize) -> Option<&cv::Mat> {
        self.pyramids.get(level)
    }

    pub fn match_model(&self, dst: &cv::Mat, mut config: MatchConfig) -> Result<Vec<Pose>> {
        ensure!(!dst.empty(), "input image is empty");
        ensure!(dst.channels() == 1, "input image must be grayscale");
        ensure!(!self.pyramids.is_empty(), "model has no pyramids");

        let template_img = &self.pyramids[0];
        ensure!(
            dst.cols() >= template_img.cols() && dst.rows() >= template_img.rows(),
            "input image is smaller than template"
        );

        let template_level = (self.pyramids.len() as i32) - 1;
        if config.level < 0 || config.level > template_level {
            config.level = template_level;
        }

        let mut pyramids = cv::Vector::<cv::Mat>::new();
        imgproc::build_pyramid_def(dst, &mut pyramids, config.level)?;

        let mut pyramid_vec = Vec::with_capacity(pyramids.len());
        for i in 0..pyramids.len() {
            pyramid_vec.push(pyramids.get(i)?);
        }

        let top = pyramid_vec
            .last()
            .ok_or_else(|| anyhow::anyhow!("empty pyramid"))?;
        let candidates = match_top_level(
            top,
            config.start_angle,
            config.span_angle,
            config.max_overlap,
            config.min_score,
            config.max_count,
            self,
            config.level as usize,
        )?;

        let mut level_matched = match_down_level(
            &pyramid_vec,
            &candidates,
            config.min_score,
            config.subpixel,
            self,
            config.level as usize,
        )?;

        let size = template_img.size()?;
        let mut rects = Vec::with_capacity(level_matched.len());
        for candidate in &level_matched {
            let base = cv::Point2f::new(candidate.pos.x as f32, candidate.pos.y as f32);
            let top_right = cv::Point2f::new(base.x + size.width as f32, base.y);
            let bottom_right =
                cv::Point2f::new(base.x + size.width as f32, base.y + size.height as f32);
            let rotate = get_rotation_matrix_2d(base, -candidate.angle)?;
            let rotated_top_right = transform_point2d(
                cv::Point2d::new(top_right.x as f64, top_right.y as f64),
                &rotate,
            )?;
            let rotated_bottom_right = transform_point2d(
                cv::Point2d::new(bottom_right.x as f64, bottom_right.y as f64),
                &rotate,
            )?;
            let rect = cv::RotatedRect::for_points(
                base,
                cv::Point2f::new(rotated_top_right.x as f32, rotated_top_right.y as f32),
                cv::Point2f::new(rotated_bottom_right.x as f32, rotated_bottom_right.y as f32),
            )?;
            rects.push(rect);
        }

        filter_overlap(&mut level_matched, &rects, config.max_overlap)?;

        let mut result = Vec::new();
        for (candidate, rect) in level_matched.iter().zip(rects.iter()) {
            if candidate.score < 0.0 {
                continue;
            }

            let center = rect.center;
            result.push(Pose {
                x: center.x,
                y: center.y,
                angle: (-candidate.angle) as f32,
                score: candidate.score as f32,
            });
        }

        result.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        Ok(result)
    }
}

pub fn match_model_multi_scale(
    template: &cv::Mat,
    dst: &cv::Mat,
    scales: &[f64],
    match_config: MatchConfig,
) -> Result<Vec<ScaledPose>> {
    ensure!(!scales.is_empty(), "scales must not be empty");
    ensure!(!template.empty(), "template image is empty");
    ensure!(template.channels() == 1, "template image must be grayscale");

    let mut results = scales
        .par_iter()
        .try_fold(Vec::new, |mut acc, &scale| -> Result<Vec<ScaledPose>> {
            ensure!(scale > 0.0, "scale must be positive");

            let scaled_template = if (scale - 1.0).abs() < TOLERANCE {
                template.clone()
            } else {
                resize_template(template, scale)?
            };

            if scaled_template.cols() > dst.cols() || scaled_template.rows() > dst.rows() {
                return Ok(acc);
            }

            let model = GrayMatchModel::train(&scaled_template, -1)?;
            let poses = model.match_model(dst, match_config.clone())?;
            acc.extend(poses.into_iter().map(|pose| ScaledPose { pose, scale }));
            Ok(acc)
        })
        .try_reduce(Vec::new, |mut a, mut b| {
            a.append(&mut b);
            Ok(a)
        })?;

    results.sort_by(|a, b| {
        b.pose
            .score
            .partial_cmp(&a.pose.score)
            .unwrap_or(Ordering::Equal)
    });
    Ok(results)
}

#[derive(Debug, Clone)]
struct Candidate {
    pos: cv::Point2d,
    angle: f64,
    score: f64,
}

impl Candidate {
    fn new(pos: cv::Point2d, angle: f64, score: f64) -> Self {
        Self { pos, angle, score }
    }
}

#[derive(Debug)]
struct Block {
    rect: cv::Rect,
    max_pos: cv::Point,
    max_score: f64,
}

impl Block {
    fn new(rect: cv::Rect, max_pos: cv::Point, max_score: f64) -> Self {
        Self {
            rect,
            max_pos,
            max_score,
        }
    }
}

struct BlockMax {
    blocks: Vec<Block>,
    score: cv::Mat,
}

impl BlockMax {
    fn new(score: cv::Mat, template_size: cv::Size) -> Result<Self> {
        let mut blocks = Vec::new();
        let block_width = template_size.width * 2;
        let block_height = template_size.height * 2;

        let score_size = score.size()?;
        let n_width = score_size.width / block_width;
        let n_height = score_size.height / block_height;
        let h_remained = score_size.width % block_width;
        let v_remained = score_size.height % block_height;

        for y in 0..n_height {
            for x in 0..n_width {
                let rect =
                    cv::Rect::new(x * block_width, y * block_height, block_width, block_height);
                let mut max_score = 0.0;
                let mut max_pos = cv::Point::new(0, 0);
                let roi = score.roi(rect)?;
                cv::min_max_loc(
                    &roi,
                    None,
                    Some(&mut max_score),
                    None,
                    Some(&mut max_pos),
                    &cv::no_array(),
                )?;
                max_pos.x += rect.x;
                max_pos.y += rect.y;
                blocks.push(Block::new(rect, max_pos, max_score));
            }
        }

        if h_remained > 0 {
            let rect = cv::Rect::new(n_width * block_width, 0, h_remained, score_size.height);
            let mut max_score = 0.0;
            let mut max_pos = cv::Point::new(0, 0);
            let roi = score.roi(rect)?;
            cv::min_max_loc(
                &roi,
                None,
                Some(&mut max_score),
                None,
                Some(&mut max_pos),
                &cv::no_array(),
            )?;
            max_pos.x += rect.x;
            max_pos.y += rect.y;
            blocks.push(Block::new(rect, max_pos, max_score));
        }

        if v_remained > 0 {
            let width = if h_remained > 0 {
                n_width * block_width
            } else {
                score_size.width
            };
            if width > 0 {
                let rect = cv::Rect::new(0, n_height * block_height, width, v_remained);
                let mut max_score = 0.0;
                let mut max_pos = cv::Point::new(0, 0);
                let roi = score.roi(rect)?;
                cv::min_max_loc(
                    &roi,
                    None,
                    Some(&mut max_score),
                    None,
                    Some(&mut max_pos),
                    &cv::no_array(),
                )?;
                max_pos.x += rect.x;
                max_pos.y += rect.y;
                blocks.push(Block::new(rect, max_pos, max_score));
            }
        }

        Ok(Self { blocks, score })
    }

    fn update(&mut self, rect: cv::Rect) -> Result<()> {
        for block in &mut self.blocks {
            if rect_intersection(block.rect, rect).is_none() {
                continue;
            }

            let mut max_score = 0.0;
            let mut max_pos = cv::Point::new(0, 0);
            let roi = self.score.roi(block.rect)?;
            cv::min_max_loc(
                &roi,
                None,
                Some(&mut max_score),
                None,
                Some(&mut max_pos),
                &cv::no_array(),
            )?;
            max_pos.x += block.rect.x;
            max_pos.y += block.rect.y;
            block.max_score = max_score;
            block.max_pos = max_pos;
        }

        Ok(())
    }

    fn max_value_loc(&self) -> Option<(f64, cv::Point)> {
        self.blocks
            .iter()
            .max_by(|a, b| {
                a.max_score
                    .partial_cmp(&b.max_score)
                    .unwrap_or(Ordering::Equal)
            })
            .map(|block| (block.max_score, block.max_pos))
    }
}

fn compute_layers(width: i32, height: i32, min_area: i32) -> i32 {
    let mut area = width * height;
    let mut layer = 0;
    while area > min_area {
        area /= 4;
        layer += 1;
    }
    layer
}

fn size_center(size: cv::Size) -> cv::Point2d {
    cv::Point2d::new(
        (size.width as f64 - 1.0) / 2.0,
        (size.height as f64 - 1.0) / 2.0,
    )
}

fn size_angle_step(size: cv::Size) -> f64 {
    let denom = size.width.max(size.height) as f64;
    (2.0 / denom).atan() * 180.0 / PI
}

fn get_rotation_matrix_2d(center: cv::Point2f, angle: f64) -> Result<cv::Mat> {
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

fn transform_point2d(point: cv::Point2d, rotate: &cv::Mat) -> Result<cv::Point2d> {
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

fn transform_with_center(
    point: cv::Point2d,
    center: cv::Point2d,
    angle: f64,
) -> Result<cv::Point2d> {
    let rotate = get_rotation_matrix_2d(cv::Point2f::new(center.x as f32, center.y as f32), angle)?;
    transform_point2d(point, &rotate)
}

fn compute_rotation_size(
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

    if angle > 0.0 && angle < 90.0 {
    } else if angle > 90.0 && angle < 180.0 {
        angle -= 90.0;
    } else if angle > 180.0 && angle < 270.0 {
        angle -= 180.0;
    } else if angle > 270.0 && angle < 360.0 {
        angle -= 270.0;
    }

    let radius = angle / 180.0 * PI;
    let dy = radius.sin();
    let dx = radius.cos();
    let width = template_size.width as f64 * dx * dy;
    let height = template_size.height as f64 * dx * dy;

    let center = size_center(dst_size);
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

fn crop_rotated_roi(
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

fn match_top_level(
    dst_top: &cv::Mat,
    start_angle: f64,
    span_angle: f64,
    max_overlap: f64,
    min_score: f64,
    max_count: usize,
    model: &GrayMatchModel,
    level: usize,
) -> Result<Vec<Candidate>> {
    let template_top = &model.pyramids[level];
    let template_size = template_top.size()?;
    let angle_step = size_angle_step(template_size);
    let center = size_center(dst_top.size()?);
    let top_score_threshold = min_score * 0.9_f64.powi(level as i32);
    let cal_max_by_block =
        size_area(dst_top.size()?) / size_area(template_size) > 500 && max_count > 10;

    let count = (span_angle / angle_step).floor() as i32 + 1;
    if count <= 0 {
        return Ok(Vec::new());
    }

    let mut candidates = (0..count)
        .into_par_iter()
        .try_fold(Vec::new, |mut acc, i| -> Result<Vec<Candidate>> {
            let angle = start_angle + angle_step * i as f64;
            let mut rotate =
                get_rotation_matrix_2d(cv::Point2f::new(center.x as f32, center.y as f32), angle)?;
            let size = compute_rotation_size(dst_top.size()?, template_size, angle, &rotate)?;

            let tx = (size.width as f64 - 1.0) / 2.0 - center.x;
            let ty = (size.height as f64 - 1.0) / 2.0 - center.y;
            *rotate.at_2d_mut::<f64>(0, 2)? += tx;
            *rotate.at_2d_mut::<f64>(1, 2)? += ty;
            let offset = cv::Point2d::new(tx, ty);

            let mut rotated = cv::Mat::default();
            imgproc::warp_affine(
                dst_top,
                &mut rotated,
                &rotate,
                size,
                imgproc::INTER_LINEAR,
                cv::BORDER_CONSTANT,
                cv::Scalar::all(model.border_color as f64),
            )?;

            let result = match_template(&rotated, model, level)?;

            if cal_max_by_block {
                let mut block = BlockMax::new(result, template_size)?;
                let (mut max_score, mut max_pos) = match block.max_value_loc() {
                    Some((score, pos)) => (score, pos),
                    None => return Ok(acc),
                };
                if max_score < top_score_threshold {
                    return Ok(acc);
                }

                acc.push(Candidate::new(
                    cv::Point2d::new(max_pos.x as f64 - offset.x, max_pos.y as f64 - offset.y),
                    angle,
                    max_score,
                ));
                let limit = max_count.saturating_add(CANDIDATE).saturating_sub(1);
                for _ in 0..limit {
                    next_max_loc_block(
                        max_pos,
                        template_size,
                        max_overlap,
                        &mut block,
                        &mut max_score,
                        &mut max_pos,
                    )?;
                    if max_score < top_score_threshold {
                        break;
                    }

                    acc.push(Candidate::new(
                        cv::Point2d::new(max_pos.x as f64 - offset.x, max_pos.y as f64 - offset.y),
                        angle,
                        max_score,
                    ));
                }
            } else {
                let mut max_score = 0.0;
                let mut max_pos = cv::Point::new(0, 0);
                cv::min_max_loc(
                    &result,
                    None,
                    Some(&mut max_score),
                    None,
                    Some(&mut max_pos),
                    &cv::no_array(),
                )?;
                if max_score < top_score_threshold {
                    return Ok(acc);
                }

                acc.push(Candidate::new(
                    cv::Point2d::new(max_pos.x as f64 - offset.x, max_pos.y as f64 - offset.y),
                    angle,
                    max_score,
                ));
                let limit = max_count.saturating_add(CANDIDATE).saturating_sub(1);
                for _ in 0..limit {
                    next_max_loc_mat(
                        &result,
                        max_pos,
                        template_size,
                        max_overlap,
                        &mut max_score,
                        &mut max_pos,
                    )?;
                    if max_score < top_score_threshold {
                        break;
                    }

                    acc.push(Candidate::new(
                        cv::Point2d::new(max_pos.x as f64 - offset.x, max_pos.y as f64 - offset.y),
                        angle,
                        max_score,
                    ));
                }
            }

            Ok(acc)
        })
        .try_reduce(Vec::new, |mut a, mut b| {
            a.append(&mut b);
            Ok(a)
        })?;

    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    Ok(candidates)
}

fn match_down_level(
    pyramids: &[cv::Mat],
    candidates: &[Candidate],
    min_score: f64,
    subpixel: bool,
    model: &GrayMatchModel,
    level: usize,
) -> Result<Vec<Candidate>> {
    let mut level_matched = candidates
        .par_iter()
        .try_fold(Vec::new, |mut acc, pose| -> Result<Vec<Candidate>> {
            let mut pose = pose.clone();
            let mut matched = true;

            for current_level in (0..level).rev() {
                let current_template = &model.pyramids[current_level];
                let tmp_size = current_template.size()?;
                let current_dst = &pyramids[current_level];
                let dst_size = current_dst.size()?;

                let current_angle_step = size_angle_step(tmp_size);
                let center = size_center(dst_size);

                let last_size = pyramids[current_level + 1].size()?;
                let last_center = size_center(last_size);
                let mut top_left = transform_with_center(pose.pos, last_center, -pose.angle)?;
                top_left.x *= 2.0;
                top_left.y *= 2.0;

                let score_threshold = min_score * 0.9_f64.powi(current_level as i32);

                let mut new_candidate = Candidate::new(cv::Point2d::new(0.0, 0.0), 0.0, 0.0);
                let mut subpixel_offset = None;

                for i in -1..=1 {
                    let angle = pose.angle + current_angle_step * i as f64;
                    let mut rotate = get_rotation_matrix_2d(
                        cv::Point2f::new(center.x as f32, center.y as f32),
                        angle,
                    )?;
                    let roi = crop_rotated_roi(current_dst, tmp_size, top_left, &mut rotate)?;

                    let result = match_template(&roi, model, current_level)?;
                    let mut max_score = 0.0;
                    let mut max_pos = cv::Point::new(0, 0);
                    cv::min_max_loc(
                        &result,
                        None,
                        Some(&mut max_score),
                        None,
                        Some(&mut max_pos),
                        &cv::no_array(),
                    )?;

                    if new_candidate.score >= max_score || max_score < score_threshold {
                        continue;
                    }

                    new_candidate = Candidate::new(
                        cv::Point2d::new(max_pos.x as f64, max_pos.y as f64),
                        angle,
                        max_score,
                    );

                    if current_level == 0 && subpixel {
                        let is_border = max_pos.x == 0
                            || max_pos.y == 0
                            || max_pos.x == result.cols() - 1
                            || max_pos.y == result.rows() - 1;
                        if !is_border {
                            let rect = cv::Rect::new(max_pos.x - 1, max_pos.y - 1, 3, 3);
                            let score_rect = result.roi(rect)?.clone_pointee();
                            subpixel_offset = Some(compute_subpixel(&score_rect)?);
                        }
                    }
                }

                if new_candidate.score < score_threshold {
                    matched = false;
                    break;
                }

                if let Some(offset) = subpixel_offset {
                    new_candidate.pos.x += offset.x as f64;
                    new_candidate.pos.y += offset.y as f64;
                }

                let mut padding_top_left =
                    transform_with_center(top_left, center, new_candidate.angle)?;
                padding_top_left.x -= 3.0;
                padding_top_left.y -= 3.0;
                new_candidate.pos.x += padding_top_left.x;
                new_candidate.pos.y += padding_top_left.y;
                pose = new_candidate;
            }

            if !matched {
                return Ok(acc);
            }

            let last_size = pyramids[0].size()?;
            let last_center = size_center(last_size);
            pose.pos = transform_with_center(pose.pos, last_center, -pose.angle)?;
            acc.push(pose);
            Ok(acc)
        })
        .try_reduce(Vec::new, |mut a, mut b| {
            a.append(&mut b);
            Ok(a)
        })?;

    level_matched.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    Ok(level_matched)
}

fn compute_subpixel(score: &cv::Mat) -> Result<cv::Point2f> {
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

fn match_template(src: &cv::Mat, model: &GrayMatchModel, level: usize) -> Result<cv::Mat> {
    let mut result = cv::Mat::default();
    imgproc::match_template(
        src,
        &model.pyramids[level],
        &mut result,
        imgproc::TM_CCORR,
        &cv::no_array(),
    )?;
    ccoeff_denominator(
        src,
        model.pyramids[level].size()?,
        &mut result,
        model.mean[level].0[0],
        model.normal[level],
        model.inv_area[level],
        model.equal1[level],
    )?;
    Ok(result)
}

fn ccoeff_denominator(
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

fn next_max_loc_block(
    pos: cv::Point,
    template_size: cv::Size,
    max_overlap: f64,
    block: &mut BlockMax,
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

    imgproc::rectangle(
        &mut block.score,
        rect_ignore,
        cv::Scalar::all(-1.0),
        imgproc::FILLED,
        imgproc::LINE_8,
        0,
    )?;

    block.update(rect_ignore)?;
    if let Some((score, pos)) = block.max_value_loc() {
        *max_score = score;
        *max_pos = pos;
    }

    Ok(())
}

fn next_max_loc_mat(
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

fn filter_overlap(
    candidates: &mut [Candidate],
    rects: &[cv::RotatedRect],
    max_overlap: f64,
) -> Result<()> {
    let size = candidates.len();
    for i in 0..size {
        if candidates[i].score < 0.0 {
            continue;
        }

        for j in (i + 1)..size {
            if candidates[j].score < 0.0 {
                continue;
            }

            let mut points = cv::Vector::<cv::Point2f>::new();
            let intersect_type =
                imgproc::rotated_rectangle_intersection(rects[i], rects[j], &mut points)?;

            match intersect_type {
                imgproc::INTERSECT_NONE => continue,
                imgproc::INTERSECT_FULL => {
                    if candidates[i].score > candidates[j].score {
                        candidates[j].score = INVALID;
                    } else {
                        candidates[i].score = INVALID;
                    }
                }
                imgproc::INTERSECT_PARTIAL => {
                    if points.len() < 2 {
                        continue;
                    }
                    let area = imgproc::contour_area_def(&points)?;
                    let rect_area = rects[i].size.width as f64 * rects[i].size.height as f64;
                    if rect_area <= 0.0 {
                        continue;
                    }
                    let overlap = area / rect_area;
                    if overlap > max_overlap {
                        if candidates[i].score > candidates[j].score {
                            candidates[j].score = INVALID;
                        } else {
                            candidates[i].score = INVALID;
                        }
                    }
                }
                _ => {}
            }
        }
    }

    Ok(())
}

fn rect_intersection(a: cv::Rect, b: cv::Rect) -> Option<cv::Rect> {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.width).min(b.x + b.width);
    let y2 = (a.y + a.height).min(b.y + b.height);
    if x2 > x1 && y2 > y1 {
        Some(cv::Rect::new(x1, y1, x2 - x1, y2 - y1))
    } else {
        None
    }
}

fn size_area(size: cv::Size) -> i32 {
    size.width * size.height
}

fn resize_template(template: &cv::Mat, scale: f64) -> Result<cv::Mat> {
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
