use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct AutoScale {
    original_size: (usize, usize),
    target_size: Option<(usize, usize)>,
    variants: BTreeMap<String, usize>,
}

impl AutoScale {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            original_size: (width, height),
            target_size: None,
            variants: BTreeMap::new(),
        }
    }

    pub fn clone_with_target_size(&self, width: usize, height: usize) -> Self {
        let mut new = self.clone();
        new.set_target_size(width, height);
        new
    }

    pub fn set_target_size(&mut self, width: usize, height: usize) {
        self.target_size = Some((width, height));
    }

    pub fn add_variant(&mut self, name: &str, size: usize) {
        self.variants.insert(name.to_string(), size);
    }

    pub fn variant_scale(&self, variant: &str) -> Option<usize> {
        let &size = self.variants.get(variant)?;
        let (target_width, target_height) = self.target_size.unwrap_or(self.original_size);
        let scale_x = target_width as f64 / self.original_size.0 as f64;
        let scale_y = target_height as f64 / self.original_size.1 as f64;
        let box_scale = scale_x.min(scale_y);
        Some((size as f64 * box_scale) as usize)
    }

    pub fn variant_scale_x(&self, variant: &str) -> Option<usize> {
        let &size = self.variants.get(variant)?;
        let (target_width, _) = self.target_size.unwrap_or(self.original_size);
        let scale_x = target_width as f64 / self.original_size.0 as f64;
        Some((size as f64 * scale_x) as usize)
    }

    pub fn variant_scale_y(&self, variant: &str) -> Option<usize> {
        let &size = self.variants.get(variant)?;
        let (_, target_height) = self.target_size.unwrap_or(self.original_size);
        let scale_y = target_height as f64 / self.original_size.1 as f64;
        Some((size as f64 * scale_y) as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_scale() {
        let mut scale = AutoScale::new(100, 100);
        scale.set_target_size(200, 200);
        scale.add_variant("small", 50);
        scale.add_variant("medium", 100);
        scale.add_variant("large", 150);

        assert_eq!(scale.variant_scale("small"), Some(100));
        assert_eq!(scale.variant_scale("medium"), Some(200));
        assert_eq!(scale.variant_scale("large"), Some(300));

        assert_eq!(scale.variant_scale_x("small"), Some(100));
        assert_eq!(scale.variant_scale_x("medium"), Some(200));
        assert_eq!(scale.variant_scale_x("large"), Some(300));

        assert_eq!(scale.variant_scale_y("small"), Some(100));
        assert_eq!(scale.variant_scale_y("medium"), Some(200));
        assert_eq!(scale.variant_scale_y("large"), Some(300));
    }
}
