use anyhow::{ensure, Result};
use opencv::core::{self, Mat, MatTraitConst};
use opencv::imgproc;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use super::core::util::{resize_mask, resize_template, TOLERANCE};
use super::core::{MatchConfig, Model, NmsConfig, ScaledPose};

#[derive(Debug, Clone)]
pub struct ModelPyramidLevel {
    pub scale: f64,
    pub model: Model,
}

#[derive(Debug, Clone)]
pub struct ModelPyramid {
    levels: Vec<ModelPyramidLevel>,
}

impl ModelPyramid {
    pub fn train(template: Mat, mut scales: Vec<f64>) -> Result<Self> {
        if scales.is_empty() {
            scales.push(1.0);
        }

        let mut levels = scales
            .par_iter()
            .try_fold(
                Vec::new,
                |mut acc, &scale| -> Result<Vec<ModelPyramidLevel>> {
                    ensure!(scale > 0.0, "scale must be positive");

                    let scaled_template = if (scale - 1.0).abs() < TOLERANCE {
                        template.clone()
                    } else {
                        resize_template(&template, scale)?
                    };

                    let model = Model::train(&scaled_template)?;
                    acc.push(ModelPyramidLevel { scale, model });
                    Ok(acc)
                },
            )
            .try_reduce(Vec::new, |mut a, mut b| {
                a.append(&mut b);
                Ok(a)
            })?;

        levels.sort_by(|a, b| {
            a.scale
                .partial_cmp(&b.scale)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(Self { levels })
    }

    pub fn train_with_mask(template: Mat, mask: Mat, mut scales: Vec<f64>) -> Result<Self> {
        if scales.is_empty() {
            scales.push(1.0);
        }
        ensure!(!mask.empty(), "mask image is empty");
        ensure!(mask.channels() == 1, "mask image must be single-channel");
        ensure!(
            mask.cols() == template.cols() && mask.rows() == template.rows(),
            "mask size must match template size"
        );

        let mut levels = scales
            .par_iter()
            .try_fold(
                Vec::new,
                |mut acc, &scale| -> Result<Vec<ModelPyramidLevel>> {
                    ensure!(scale > 0.0, "scale must be positive");

                    let (scaled_template, scaled_mask) = if (scale - 1.0).abs() < TOLERANCE {
                        (template.clone(), mask.clone())
                    } else {
                        (resize_template(&template, scale)?, resize_mask(&mask, scale)?)
                    };

                    let model = Model::train_with_mask(&scaled_template, &scaled_mask)?;
                    acc.push(ModelPyramidLevel { scale, model });
                    Ok(acc)
                },
            )
            .try_reduce(Vec::new, |mut a, mut b| {
                a.append(&mut b);
                Ok(a)
            })?;

        levels.sort_by(|a, b| {
            a.scale
                .partial_cmp(&b.scale)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(Self { levels })
    }

    pub fn levels(&self) -> &[ModelPyramidLevel] {
        &self.levels
    }

    pub fn find_matching_points(
        &self,
        input: &Mat,
        match_config: MatchConfig,
    ) -> Result<Vec<ScaledPose>> {
        let mut results = self
            .levels
            .par_iter()
            .try_fold(Vec::new, |mut acc, level| -> Result<Vec<ScaledPose>> {
                let poses = level.model.match_model(input, match_config.clone())?;
                acc.extend(poses.into_iter().map(|pose| ScaledPose {
                    pose,
                    scale: level.scale,
                }));
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
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(results)
    }

    pub fn find_best_matches(
        &self,
        input: &Mat,
        match_config: MatchConfig,
        nms_config: NmsConfig,
    ) -> Result<Vec<ScaledPose>> {
        let results = self.find_matching_points(input, match_config)?;
        if results.is_empty() {
            return Ok(results);
        }

        ScaledPose::nms_filter(&results, nms_config)
    }

    pub fn find_best_matches_adaptive(
        &self,
        input: &Mat,
        match_config: MatchConfig,
        nms_config: NmsConfig,
        adapt_scales: Vec<f64>,
    ) -> Result<Vec<ScaledPose>> {
        ensure!(!adapt_scales.is_empty(), "adapt_scales must not be empty");

        let mut bootstrap_config = match_config.clone();
        bootstrap_config.max_count = 1;
        let bootstrap_matches = self.find_matching_points(input, bootstrap_config)?;
        if bootstrap_matches.is_empty() {
            return Ok(bootstrap_matches);
        }

        let best = bootstrap_matches
            .iter()
            .max_by(|a, b| {
                a.pose
                    .score
                    .partial_cmp(&b.pose.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| anyhow::anyhow!("no bootstrap matches found"))?;
        let best_scale = best.scale;

        let level = self
            .levels
            .iter()
            .min_by(|a, b| {
                let da = (a.scale - best_scale).abs();
                let db = (b.scale - best_scale).abs();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| anyhow::anyhow!("no matching pyramid levels"))?;

        let template_from_source = crop_template_from_pose(input, &best.pose)?;
        let refined_pyramid = if let Some(mask) = level.model.template_mask() {
            ModelPyramid::train_with_mask(
                template_from_source,
                mask.clone(),
                adapt_scales,
            )?
        } else {
            ModelPyramid::train(template_from_source, adapt_scales)?
        };

        let mut refined =
            refined_pyramid.find_best_matches(input, match_config, nms_config)?;
        for pose in &mut refined {
            pose.scale *= best_scale;
        }
        Ok(refined)
    }
}

impl ModelPyramid {
    pub fn from_model(model: &Model, scales: Vec<f64>) -> Result<Self> {
        Self::train(model.template().clone(), scales)
    }
}

pub fn match_model_multi_scale(
    template: &Mat,
    dst: &Mat,
    scales: &[f64],
    match_config: MatchConfig,
) -> Result<Vec<ScaledPose>> {
    ensure!(!scales.is_empty(), "scales must not be empty");
    ensure!(!template.empty(), "template image is empty");
    ensure!(template.channels() == 1, "template image must be grayscale");

    let pyramid = ModelPyramid::train(template.clone(), scales.to_vec())?;
    pyramid.find_matching_points(dst, match_config)
}

pub fn match_model_multi_scale_with_mask(
    template: &Mat,
    mask: &Mat,
    dst: &Mat,
    scales: &[f64],
    match_config: MatchConfig,
) -> Result<Vec<ScaledPose>> {
    ensure!(!scales.is_empty(), "scales must not be empty");
    ensure!(!template.empty(), "template image is empty");
    ensure!(template.channels() == 1, "template image must be grayscale");

    let pyramid = ModelPyramid::train_with_mask(template.clone(), mask.clone(), scales.to_vec())?;
    pyramid.find_matching_points(dst, match_config)
}

pub fn match_model_multi_scale_nms(
    template: &Mat,
    dst: &Mat,
    scales: &[f64],
    match_config: MatchConfig,
    nms_config: NmsConfig,
) -> Result<Vec<ScaledPose>> {
    let results = match_model_multi_scale(template, dst, scales, match_config)?;
    if results.is_empty() {
        return Ok(results);
    }

    ScaledPose::nms_filter(&results, nms_config)
}

fn crop_template_from_pose(src: &Mat, pose: &super::core::Pose) -> Result<Mat> {
    let size = src.size()?;
    let width = pose.width.round() as i32;
    let height = pose.height.round() as i32;
    ensure!(width > 0 && height > 0, "pose size is invalid");

    let mut rect = core::Rect::new(
        (pose.x - pose.width / 2.0).round() as i32,
        (pose.y - pose.height / 2.0).round() as i32,
        width,
        height,
    );
    if rect.x < 0 {
        rect.width -= -rect.x;
        rect.x = 0;
    }
    if rect.y < 0 {
        rect.height -= -rect.y;
        rect.y = 0;
    }
    if rect.x + rect.width > size.width {
        rect.width = size.width - rect.x;
    }
    if rect.y + rect.height > size.height {
        rect.height = size.height - rect.y;
    }
    ensure!(rect.width > 0 && rect.height > 0, "cropped template is out of bounds");

    let center = core::Point2f::new(pose.x, pose.y);
    let rotate = imgproc::get_rotation_matrix_2d(center, -pose.angle as f64, 1.0)?;
    let mut rotated = core::Mat::default();
    imgproc::warp_affine(
        src,
        &mut rotated,
        &rotate,
        size,
        imgproc::INTER_LINEAR,
        core::BORDER_REPLICATE,
        core::Scalar::default(),
    )?;

    let roi = rotated.roi(rect)?;
    Ok(roi.clone_pointee())
}

pub fn match_model_multi_scale_nms_with_mask(
    template: &Mat,
    mask: &Mat,
    dst: &Mat,
    scales: &[f64],
    match_config: MatchConfig,
    nms_config: NmsConfig,
) -> Result<Vec<ScaledPose>> {
    let results =
        match_model_multi_scale_with_mask(template, mask, dst, scales, match_config)?;
    if results.is_empty() {
        return Ok(results);
    }

    ScaledPose::nms_filter(&results, nms_config)
}
