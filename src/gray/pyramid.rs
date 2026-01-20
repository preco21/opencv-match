use anyhow::{ensure, Result};
use opencv::core::{Mat, MatTraitConst};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use super::core::util::{resize_template, TOLERANCE};
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
