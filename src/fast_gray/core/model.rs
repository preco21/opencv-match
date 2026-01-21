use anyhow::{bail, ensure, Result};
use opencv::core::{self as cv, MatTraitConst};
use opencv::imgproc;
use opencv::prelude::*;
use rayon::prelude::*;
use std::cmp::Ordering;

use super::internal::{compute_layers, size_area, BlockMax, Candidate, Pose, CANDIDATE, MIN_AREA};
use super::util::{
    ccoeff_denominator, compute_rotation_size, compute_subpixel, crop_rotated_roi,
    get_rotation_matrix_2d, next_max_loc_mat, size_angle_step, size_center, transform_point2d,
    transform_with_center,
};

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

#[derive(Debug, Clone, Copy)]
pub struct NmsConfig {
    /// The threshold for the Intersection over Union (IoU) to use for non-maximum suppression.
    pub iou_threshold: f64,
    /// The threshold for the score to use for non-maximum suppression.
    pub score_threshold: f64,
}

impl Default for NmsConfig {
    fn default() -> Self {
        Self {
            iou_threshold: 0.0,
            score_threshold: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    pyramids: Vec<cv::Mat>,
    mean: Vec<cv::Scalar>,
    normal: Vec<f64>,
    inv_area: Vec<f64>,
    equal1: Vec<bool>,
    border_color: u8,
    mask_pyramids: Option<Vec<cv::Mat>>,
}

impl Model {
    pub fn train(template: &cv::Mat) -> Result<Self> {
        Self::new(template, -1)
    }

    pub fn train_with_mask(template: &cv::Mat, mask: &cv::Mat) -> Result<Self> {
        Self::new_with_mask(template, mask, -1)
    }

    pub fn new(template: &cv::Mat, level: i32) -> Result<Self> {
        Self::new_with_optional_mask(template, level, None)
    }

    pub fn new_with_mask(template: &cv::Mat, mask: &cv::Mat, level: i32) -> Result<Self> {
        Self::new_with_optional_mask(template, level, Some(mask))
    }

    fn new_with_optional_mask(
        template: &cv::Mat,
        level: i32,
        mask: Option<&cv::Mat>,
    ) -> Result<Self> {
        ensure!(!template.empty(), "template image is empty");
        ensure!(template.channels() == 1, "template image must be grayscale");
        if let Some(mask) = mask {
            ensure!(!mask.empty(), "mask image is empty");
            ensure!(mask.channels() == 1, "mask image must be single-channel");
            ensure!(
                mask.cols() == template.cols() && mask.rows() == template.rows(),
                "mask size must match template size"
            );
        }

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
        let base_mask = if let Some(mask) = mask {
            Some(normalize_mask(mask)?)
        } else {
            None
        };
        let mut mask_pyramids = if base_mask.is_some() {
            Some(Vec::with_capacity(pyramids.len()))
        } else {
            None
        };
        let mut model = Model {
            pyramids: Vec::with_capacity(pyramids.len()),
            mean: Vec::with_capacity(pyramids.len()),
            normal: Vec::with_capacity(pyramids.len()),
            inv_area: Vec::with_capacity(pyramids.len()),
            equal1: Vec::with_capacity(pyramids.len()),
            border_color: 0,
            mask_pyramids: None,
        };

        let mean_value = if let Some(mask) = &base_mask {
            cv::mean(template, mask)?.0[0]
        } else {
            cv::mean(template, &cv::no_array())?.0[0]
        };
        model.border_color = if mean_value < 128.0 { 255 } else { 0 };

        for i in 0..pyramids.len() {
            let pyramid = pyramids.get(i)?;
            let mask = match (&base_mask, &mut mask_pyramids) {
                (Some(base_mask), Some(mask_pyramids)) => {
                    let mut resized = cv::Mat::default();
                    imgproc::resize(
                        base_mask,
                        &mut resized,
                        pyramid.size()?,
                        0.0,
                        0.0,
                        imgproc::INTER_NEAREST,
                    )?;
                    mask_pyramids.push(resized);
                    mask_pyramids.last().map(|mask| mask.clone())
                }
                _ => None,
            };

            let area = (pyramid.rows() * pyramid.cols()) as f64;
            let inv_area: f64 = 1.0 / area;

            let mut mean = cv::Scalar::default();
            let mut stddev = cv::Scalar::default();
            if let Some(mask) = mask.as_ref() {
                cv::mean_std_dev(&pyramid, &mut mean, &mut stddev, mask)?;
            } else {
                cv::mean_std_dev(&pyramid, &mut mean, &mut stddev, &cv::no_array())?;
            }

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

        model.mask_pyramids = mask_pyramids;
        Ok(model)
    }

    pub fn pyramid_levels(&self) -> usize {
        self.pyramids.len()
    }

    pub fn pyramid(&self, level: usize) -> Option<&cv::Mat> {
        self.pyramids.get(level)
    }

    pub fn template(&self) -> &cv::Mat {
        &self.pyramids[0]
    }

    pub fn template_mask(&self) -> Option<&cv::Mat> {
        self.mask_pyramids.as_ref().and_then(|masks| masks.get(0))
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
        let candidates = self.find_candidates_pass(
            top,
            config.start_angle,
            config.span_angle,
            config.max_overlap,
            config.min_score,
            config.max_count,
            config.level as usize,
        )?;

        let mut level_matched = self.refine_candidates_pass(
            &pyramid_vec,
            &candidates,
            config.min_score,
            config.subpixel,
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

        Candidate::filter_overlap(&mut level_matched, &rects, config.max_overlap)?;

        let mut result = Vec::new();
        for candidate in &level_matched {
            if candidate.score < 0.0 {
                continue;
            }

            let top_left = candidate.pos;
            let angle = normalize_pose_angle(-candidate.angle);
            result.push(Pose {
                x: top_left.x as f32,
                y: top_left.y as f32,
                width: size.width as f32,
                height: size.height as f32,
                angle: angle as f32,
                score: candidate.score as f32,
            });
        }

        result.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        Ok(result)
    }

    pub fn match_model_nms(
        &self,
        dst: &cv::Mat,
        match_config: MatchConfig,
        nms_config: NmsConfig,
    ) -> Result<Vec<Pose>> {
        let poses = self.match_model(dst, match_config)?;
        if poses.is_empty() {
            return Ok(poses);
        }

        Pose::nms_filter(&poses, nms_config)
    }

    fn match_template(&self, src: &cv::Mat, level: usize) -> Result<cv::Mat> {
        let mut result = cv::Mat::default();
        if let Some(masks) = &self.mask_pyramids {
            imgproc::match_template(
                src,
                &self.pyramids[level],
                &mut result,
                imgproc::TM_CCOEFF_NORMED,
                &masks[level],
            )?;
            return Ok(result);
        }
        imgproc::match_template(
            src,
            &self.pyramids[level],
            &mut result,
            imgproc::TM_CCORR,
            &cv::no_array(),
        )?;
        ccoeff_denominator(
            src,
            self.pyramids[level].size()?,
            &mut result,
            self.mean[level].0[0],
            self.normal[level],
            self.inv_area[level],
            self.equal1[level],
        )?;
        Ok(result)
    }

    fn find_candidates_pass(
        &self,
        dst_top: &cv::Mat,
        start_angle: f64,
        span_angle: f64,
        max_overlap: f64,
        min_score: f64,
        max_count: usize,
        level: usize,
    ) -> Result<Vec<Candidate>> {
        let template_top = &self.pyramids[level];
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
                let mut rotate = get_rotation_matrix_2d(
                    cv::Point2f::new(center.x as f32, center.y as f32),
                    angle,
                )?;
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
                    cv::Scalar::all(self.border_color as f64),
                )?;

                let result = self.match_template(&rotated, level)?;

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
                        block_update_next_max_loc(
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
                            cv::Point2d::new(
                                max_pos.x as f64 - offset.x,
                                max_pos.y as f64 - offset.y,
                            ),
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
                            cv::Point2d::new(
                                max_pos.x as f64 - offset.x,
                                max_pos.y as f64 - offset.y,
                            ),
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

    fn refine_candidates_pass(
        &self,
        pyramids: &[cv::Mat],
        candidates: &[Candidate],
        min_score: f64,
        subpixel: bool,
        level: usize,
    ) -> Result<Vec<Candidate>> {
        let mut level_matched = candidates
            .par_iter()
            .try_fold(Vec::new, |mut acc, pose| -> Result<Vec<Candidate>> {
                let mut pose = pose.clone();
                let mut matched = true;

                for current_level in (0..level).rev() {
                    let current_template = &self.pyramids[current_level];
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

                        let result = self.match_template(&roi, current_level)?;
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
}

fn normalize_pose_angle(angle: f64) -> f64 {
    let mut angle = angle % 360.0;
    if angle <= -180.0 {
        angle += 360.0;
    } else if angle > 180.0 {
        angle -= 360.0;
    }
    if angle.abs() >= 180.0 {
        0.0
    } else {
        angle
    }
}

fn normalize_mask(mask: &cv::Mat) -> Result<cv::Mat> {
    let mask_u8 = if mask.depth() == cv::CV_8U {
        mask.clone()
    } else {
        let mut converted = cv::Mat::default();
        mask.convert_to(&mut converted, cv::CV_8U, 1.0, 0.0)?;
        converted
    };

    let mut binary = cv::Mat::default();
    imgproc::threshold(
        &mask_u8,
        &mut binary,
        0.0,
        255.0,
        imgproc::THRESH_BINARY,
    )?;
    Ok(binary)
}

fn block_update_next_max_loc(
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
