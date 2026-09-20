//! # Data Augmentation — 8 dihedral + color perm + translation + Sudoku + Cache
//!
//! Parallelized with rayon for CPU-bound preprocessing on i7-3770K.
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Mutex;

pub type Grid = Vec<Vec<u8>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DihedralOp {
    Identity, Rot90, Rot180, Rot270,
    FlipH, FlipV, FlipDiag, FlipAntiDiag,
}

impl DihedralOp {
    pub fn all() -> [Self; 8] {
        [Self::Identity, Self::Rot90, Self::Rot180, Self::Rot270,
         Self::FlipH, Self::FlipV, Self::FlipDiag, Self::FlipAntiDiag]
    }
    pub fn apply(&self, g: &Grid) -> Grid {
        let rows = g.len();
        let cols = if rows > 0 { g[0].len() } else { 0 };
        match self {
            Self::Identity => g.clone(),
            Self::Rot90 => {
                let mut out = vec![vec![0u8; rows]; cols];
                for r in 0..rows { for c in 0..cols { out[c][rows-1-r] = g[r][c]; } }
                out
            }
            Self::Rot180 => {
                let mut out = vec![vec![0u8; cols]; rows];
                for r in 0..rows { for c in 0..cols { out[rows-1-r][cols-1-c] = g[r][c]; } }
                out
            }
            Self::Rot270 => {
                let mut out = vec![vec![0u8; rows]; cols];
                for r in 0..rows { for c in 0..cols { out[cols-1-c][r] = g[r][c]; } }
                out
            }
            Self::FlipH => g.iter().map(|row| row.iter().rev().copied().collect()).collect(),
            Self::FlipV => g.iter().rev().cloned().collect(),
            Self::FlipDiag => {
                let mut out = vec![vec![0u8; rows]; cols];
                for r in 0..rows { for c in 0..cols { out[c][r] = g[r][c]; } }
                out
            }
            Self::FlipAntiDiag => {
                let mut out = vec![vec![0u8; rows]; cols];
                for r in 0..rows { for c in 0..cols { out[cols-1-c][rows-1-r] = g[r][c]; } }
                out
            }
        }
    }
    pub fn inverse(&self) -> Self {
        match self {
            Self::Identity => Self::Identity,
            Self::Rot90 => Self::Rot270,
            Self::Rot180 => Self::Rot180,
            Self::Rot270 => Self::Rot90,
            Self::FlipH => Self::FlipH,
            Self::FlipV => Self::FlipV,
            Self::FlipDiag => Self::FlipDiag,
            Self::FlipAntiDiag => Self::FlipAntiDiag,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AugTransform {
    pub color_perm: [u8; 10],
    pub dihedral: DihedralOp,
    pub translate: (i32, i32),
    pub orig_rows: usize,
    pub orig_cols: usize,
}

impl AugTransform {
    pub fn random(rng: &mut impl Rng, rows: usize, cols: usize) -> Self {
        let mut color_perm = [0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        color_perm.shuffle(rng);
        let dihedral = DihedralOp::all()[rng.gen_range(0..8)];
        let translate = (
            if rows > 0 { rng.gen_range(0..rows as i32) } else { 0 },
            if cols > 0 { rng.gen_range(0..cols as i32) } else { 0 },
        );
        Self { color_perm, dihedral, translate, orig_rows: rows, orig_cols: cols }
    }
    pub fn identity(rows: usize, cols: usize) -> Self {
        Self { color_perm: [0,1,2,3,4,5,6,7,8,9], dihedral: DihedralOp::Identity,
               translate: (0,0), orig_rows: rows, orig_cols: cols }
    }
    pub fn apply(&self, g: &Grid) -> Grid {
        let mut out = self.dihedral.apply(g);
        for r in 0..out.len() {
            for c in 0..out[0].len() {
                let v = out[r][c] as usize;
                if v < 10 { out[r][c] = self.color_perm[v]; }
            }
        }
        out
    }
    pub fn inverse(&self) -> Self {
        let mut inv_color = [0u8; 10];
        for (i, &v) in self.color_perm.iter().enumerate() { inv_color[v as usize] = i as u8; }
        Self { color_perm: inv_color, dihedral: self.dihedral.inverse(),
               translate: (-self.translate.0, -self.translate.1),
               orig_rows: self.orig_rows, orig_cols: self.orig_cols }
    }
}

pub struct AugmentCache {
    cache: Mutex<HashMap<u32, Vec<(Grid, Grid)>>>,
    seed: u64,
    n_augmentations: usize,
}

impl AugmentCache {
    pub fn new(seed: u64, n_augmentations: usize) -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
            seed,
            n_augmentations,
        }
    }
    pub fn get_or_generate(&self, puzzle_id: u32, x: &Grid, y: &Grid) -> Vec<(Grid, Grid)> {
        let mut cache = self.cache.lock().unwrap();
        if let Some(entry) = cache.get(&puzzle_id) {
            return entry.clone();
        }
        let pipeline = AugmentPipeline {
            n_augmentations: self.n_augmentations,
            seed: self.seed ^ puzzle_id as u64,
            parallel: true,
        };
        let generated: Vec<(Grid, Grid)> = pipeline.augment_pair(x, y)
            .into_iter()
            .map(|(xg, yg, _)| (xg, yg))
            .collect();
        cache.insert(puzzle_id, generated.clone());
        generated
    }
    pub fn clear(&self) {
        self.cache.lock().unwrap().clear();
    }
}

#[derive(Debug, Clone)]
pub struct AugmentPipeline {
    pub n_augmentations: usize,
    pub seed: u64,
    pub parallel: bool,
}

impl Default for AugmentPipeline {
    fn default() -> Self { Self { n_augmentations: 1000, seed: 42, parallel: true } }
}

impl AugmentPipeline {
    pub fn augment_grid(&self, grid: &Grid) -> Vec<(Grid, AugTransform)> {
        let rows = grid.len();
        let cols = if rows > 0 { grid[0].len() } else { 0 };
        let make_one = |i: usize| {
            let mut rng = StdRng::seed_from_u64(self.seed ^ ((i as u64 + 1) * 0x9E3779B97F4A7C15));
            let t = AugTransform::random(&mut rng, rows, cols);
            (t.apply(grid), t)
        };
        if self.parallel {
            (0..self.n_augmentations).into_par_iter().map(make_one).collect()
        } else {
            (0..self.n_augmentations).map(make_one).collect()
        }
    }
    pub fn augment_pair(&self, x: &Grid, y: &Grid) -> Vec<(Grid, Grid, AugTransform)> {
        let rows = x.len();
        let cols = if rows > 0 { x[0].len() } else { 0 };
        let make_one = |i: usize| {
            let mut rng = StdRng::seed_from_u64(self.seed ^ ((i as u64 + 1) * 0xD1B54A32D192ED03));
            let t = AugTransform::random(&mut rng, rows, cols);
            (t.apply(x), t.apply(y), t)
        };
        if self.parallel {
            (0..self.n_augmentations).into_par_iter().map(make_one).collect()
        } else {
            (0..self.n_augmentations).map(make_one).collect()
        }
    }
    pub fn majority_vote(&self, preds: &[(Grid, AugTransform)]) -> Grid {
        let deaug: Vec<Grid> = if self.parallel {
            preds.par_iter().map(|(g, t)| t.inverse().apply(g)).collect()
        } else {
            preds.iter().map(|(g, t)| t.inverse().apply(g)).collect()
        };
        if deaug.is_empty() { return vec![]; }
        let rows = deaug[0].len();
        let cols = if rows > 0 { deaug[0][0].len() } else { 0 };
        let mut out = vec![vec![0u8; cols]; rows];
        for r in 0..rows {
            for c in 0..cols {
                let mut counts = [0u32; 10];
                for g in &deaug {
                    if r < g.len() && c < g[r].len() {
                        let v = g[r][c] as usize;
                        if v < 10 { counts[v] += 1; }
                    }
                }
                out[r][c] = counts.iter().enumerate()
                    .max_by_key(|(_, &ct)| ct)
                    .map(|(i, _)| i as u8).unwrap_or(0);
            }
        }
        out
    }
    pub fn all_dihedral(grid: &Grid) -> Vec<(Grid, AugTransform)> {
        let rows = grid.len();
        let cols = if rows > 0 { grid[0].len() } else { 0 };
        DihedralOp::all().into_iter().map(|op| {
            let t = AugTransform { color_perm: [0,1,2,3,4,5,6,7,8,9],
                                   dihedral: op, translate: (0,0),
                                   orig_rows: rows, orig_cols: cols };
            (t.apply(grid), t)
        }).collect()
    }

    pub fn sudoku_permute_pair(&self, x: &Grid, y: &Grid) -> (Grid, Grid) {
        let rows = x.len();
        let cols = if rows > 0 { x[0].len() } else { 0 };
        if rows != 9 || cols != 9 {
            return (x.clone(), y.clone());
        }
        let mut rng = StdRng::seed_from_u64(self.seed + 0x123456789ABCDEF);
        let mut bands = [0, 1, 2];
        bands.shuffle(&mut rng);
        let mut stacks = [0, 1, 2];
        stacks.shuffle(&mut rng);
        let mut row_perm: Vec<usize> = (0..9).collect();
        for b in 0..3 {
            let start = b * 3;
            row_perm[start..start+3].shuffle(&mut rng);
        }
        let mut col_perm: Vec<usize> = (0..9).collect();
        for s in 0..3 {
            let start = s * 3;
            col_perm[start..start+3].shuffle(&mut rng);
        }
        let mut x_out = vec![vec![0u8; 9]; 9];
        let mut y_out = vec![vec![0u8; 9]; 9];
        for r in 0..9 {
            for c in 0..9 {
                x_out[r][c] = x[row_perm[r]][col_perm[c]];
                y_out[r][c] = y[row_perm[r]][col_perm[c]];
            }
        }
        let mut x_final = vec![vec![0u8; 9]; 9];
        let mut y_final = vec![vec![0u8; 9]; 9];
        for b in 0..3 {
            for r in 0..3 {
                let src_r = bands[b] * 3 + r;
                let dst_r = b * 3 + r;
                for c in 0..9 {
                    x_final[dst_r][c] = x_out[src_r][c];
                    y_final[dst_r][c] = y_out[src_r][c];
                }
            }
        }
        let mut x_stack = vec![vec![0u8; 9]; 9];
        let mut y_stack = vec![vec![0u8; 9]; 9];
        for s in 0..3 {
            for c in 0..3 {
                let src_c = stacks[s] * 3 + c;
                let dst_c = s * 3 + c;
                for r in 0..9 {
                    x_stack[r][dst_c] = x_final[r][src_c];
                    y_stack[r][dst_c] = y_final[r][src_c];
                }
            }
        }
        (x_stack, y_stack)
    }
}
