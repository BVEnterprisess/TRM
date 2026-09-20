//! Dataset loading for ARC-AGI, Sudoku, Maze
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use anyhow::{Context, Result};
use serde::Deserialize;
use crate::augment::Grid;

#[derive(Debug, Clone)]
pub struct TokenizedExample {
    pub puzzle_id: u32,
    pub x_tokens: Vec<u32>,
    pub y_tokens: Vec<u32>,
    pub rows: usize,
    pub cols: usize,
    pub x_seq_len: usize,
    pub y_seq_len: usize,
}

#[derive(Debug, Clone)]
pub struct BucketedBatch {
    pub rows: usize,
    pub cols: usize,
    pub examples: Vec<TokenizedExample>,
}

#[derive(Deserialize)]
struct ArcTaskJson { train: Vec<ArcPairJson>, test: Vec<ArcPairJson> }
#[derive(Deserialize)]
struct ArcPairJson { input: Vec<Vec<u8>>, output: Vec<Vec<u8>> }

pub struct ArcTask {
    pub train_pairs: Vec<(Grid, Grid)>,
    pub test_inputs: Vec<Grid>,
    pub test_outputs: Vec<Grid>,
    pub puzzle_id: u32,
}

pub fn tokenize_grid(grid: &Grid) -> Vec<u32> {
    grid.iter().flat_map(|row| row.iter().map(|&v| v as u32)).collect()
}
pub fn detokenize_grid(tokens: &[u32], rows: usize, cols: usize) -> Grid {
    let mut out = vec![vec![0u8; cols]; rows];
    for r in 0..rows { for c in 0..cols {
        let idx = r * cols + c;
        if idx < tokens.len() { out[r][c] = tokens[idx] as u8; }
    }}
    out
}
pub fn pad_tokens(tokens: &[u32], len: usize, pad: u32) -> Vec<u32> {
    let mut out = vec![pad; len];
    let n = tokens.len().min(len);
    out[..n].copy_from_slice(&tokens[..n]);
    out
}

pub fn ensure_sample_tasks(dir: &Path) -> Result<()> {
    if !dir.exists() {
        fs::create_dir_all(dir)?;
    }
    let has_json = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .any(|e| e.path().extension().map(|ext| ext == "json").unwrap_or(false));

    if !has_json {
        log::info!("No ARC tasks found in {}; auto-seeding sample reasoning puzzles", dir.display());
        // Seed 4 canonical ARC-style geometric reasoning tasks
        let sample1 = r#"{
            "train": [
                {"input": [[0,1,0],[1,1,1],[0,1,0]], "output": [[1,1,1],[1,0,1],[1,1,1]]},
                {"input": [[0,2,0],[2,2,2],[0,2,0]], "output": [[2,2,2],[2,0,2],[2,2,2]]}
            ],
            "test": [
                {"input": [[0,3,0],[3,3,3],[0,3,0]], "output": [[3,3,3],[3,0,3],[3,3,3]]}
            ]
        }"#;
        let sample2 = r#"{
            "train": [
                {"input": [[1,0,0],[0,1,0],[0,0,1]], "output": [[0,0,1],[0,1,0],[1,0,0]]},
                {"input": [[2,0,0],[0,2,0],[0,0,2]], "output": [[0,0,2],[0,2,0],[2,0,0]]}
            ],
            "test": [
                {"input": [[4,0,0],[0,4,0],[0,0,4]], "output": [[0,0,4],[0,4,0],[4,0,0]]}
            ]
        }"#;
        let sample3 = r#"{
            "train": [
                {"input": [[1,2],[3,4]], "output": [[1,1,2,2],[1,1,2,2],[3,3,4,4],[3,3,4,4]]}
            ],
            "test": [
                {"input": [[5,6],[7,8]], "output": [[5,5,6,6],[5,5,6,6],[7,7,8,8],[7,7,8,8]]}
            ]
        }"#;
        let sample4 = r#"{
            "train": [
                {"input": [[0,0,1],[0,1,0],[1,0,0]], "output": [[1,1,1],[1,1,1],[1,1,1]]}
            ],
            "test": [
                {"input": [[0,0,2],[0,2,0],[2,0,0]], "output": [[2,2,2],[2,2,2],[2,2,2]]}
            ]
        }"#;
        fs::write(dir.join("sample_invert_cross.json"), sample1)?;
        fs::write(dir.join("sample_flip_diagonal.json"), sample2)?;
        fs::write(dir.join("sample_scale_2x.json"), sample3)?;
        fs::write(dir.join("sample_fill_bounding.json"), sample4)?;
    }
    Ok(())
}

pub fn load_arc_tasks(dir: &Path) -> Result<Vec<(String, ArcTask)>> {
    let _ = ensure_sample_tasks(dir);
    let mut out = Vec::new();
    let entries: Vec<_> = fs::read_dir(dir)
        .with_context(|| format!("cannot read ARC directory {}", dir.display()))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut sorted_entries = entries;
    sorted_entries.sort_by_key(|e| e.path());
    for (idx, entry) in sorted_entries.into_iter().enumerate() {
        let path = entry.path();
        if path.extension().map(|e| e == "json").unwrap_or(false) {
            let txt = fs::read_to_string(&path)?;
            let raw: ArcTaskJson = serde_json::from_str(&txt)
                .with_context(|| format!("bad ARC JSON: {}", path.display()))?;
            let task = ArcTask {
                train_pairs: raw.train.iter().map(|p| (p.input.clone(), p.output.clone())).collect(),
                test_inputs: raw.test.iter().map(|p| p.input.clone()).collect(),
                test_outputs: raw.test.iter().map(|p| p.output.clone()).collect(),
                puzzle_id: idx as u32,
            };
            let stem = path.file_stem().map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| format!("task_{idx:04}"));
            out.push((stem, task));
        }
    }
    Ok(out)
}

pub fn arc_training_examples(tasks: &[(String, ArcTask)]) -> Vec<TokenizedExample> {
    let mut out = Vec::new();
    for (_, task) in tasks {
        for (x, y) in &task.train_pairs {
            let rows = x.len();
            let cols = if rows > 0 { x[0].len() } else { 0 };
            let x_tok = tokenize_grid(x);
            let y_tok = tokenize_grid(y);
            out.push(TokenizedExample {
                puzzle_id: task.puzzle_id,
                x_tokens: x_tok.clone(),
                y_tokens: y_tok.clone(),
                rows, cols,
                x_seq_len: x_tok.len(),
                y_seq_len: y_tok.len(),
            });
        }
    }
    out
}

pub fn bucket_by_shape(examples: Vec<TokenizedExample>) -> Vec<BucketedBatch> {
    let mut map: BTreeMap<(usize, usize), Vec<TokenizedExample>> = BTreeMap::new();
    for ex in examples { map.entry((ex.rows, ex.cols)).or_default().push(ex); }
    map.into_iter().map(|((rows, cols), examples)| BucketedBatch { rows, cols, examples }).collect()
}

// Sudoku
pub fn load_sudoku_txt(path: &Path) -> Result<Vec<TokenizedExample>> {
    if !path.exists() {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let sample = "\
003020600900305001001806400008102900700000008006708200002609500800203009005010300,483921657967345821251876493548132976729564138136798245372689514814253769695417382
200080300060070084030500209000105408000000000402706000301007040720040060004010003,264185379579273184138564297793512468851946732426738915315697842972841653684329571
";
        fs::write(path, sample)?;
    }
    let text = fs::read_to_string(path)?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() { continue; }
        let parts: Vec<_> = line.split(',').collect();
        if parts.len() != 2 { continue; }
        let x_tokens: Vec<u32> = parts[0].chars().filter_map(|c| c.to_digit(10)).collect();
        let y_tokens: Vec<u32> = parts[1].chars().filter_map(|c| c.to_digit(10)).collect();
        if x_tokens.len() == 81 && y_tokens.len() == 81 {
            out.push(TokenizedExample {
                puzzle_id: i as u32,
                x_tokens: x_tokens.clone(),
                y_tokens: y_tokens.clone(),
                rows: 9, cols: 9,
                x_seq_len: x_tokens.len(),
                y_seq_len: y_tokens.len(),
            });
        }
    }
    Ok(out)
}

// Maze
pub fn load_maze_dir(dir: &Path) -> Result<Vec<TokenizedExample>> {
    if !dir.exists() {
        fs::create_dir_all(dir)?;
    }
    let entries = fs::read_dir(dir)?;
    let mut files = Vec::new();
    for entry in entries.filter_map(|e| e.ok()) {
        if entry.path().is_file() {
            files.push(entry.path());
        }
    }
    if files.is_empty() {
        let sample_maze = "#####\n#S..#\n#.#.#\n#..E#\n#####\n";
        let sample_file = dir.join("sample_maze_001.txt");
        fs::write(&sample_file, sample_maze)?;
        files.push(sample_file);
    }
    files.sort();
    let mut out = Vec::new();
    for (i, path) in files.into_iter().enumerate() {
        let txt = fs::read_to_string(&path)?;
        let lines: Vec<_> = txt.lines().collect();
        if lines.is_empty() { continue; }
        let rows = lines.len();
        let cols = lines[0].chars().count();
        let mut x = Vec::with_capacity(rows * cols);
        let mut y = Vec::with_capacity(rows * cols);
        for line in lines {
            for ch in line.chars() {
                let tok = match ch { '#' => 0u32, '.' => 1, 'S' => 2, 'E' => 3, '*' => 4, _ => 1 };
                x.push(tok); y.push(tok);
            }
        }
        out.push(TokenizedExample {
            puzzle_id: i as u32,
            x_tokens: x.clone(),
            y_tokens: y.clone(),
            rows, cols,
            x_seq_len: x.len(),
            y_seq_len: y.len(),
        });
    }
    Ok(out)
}

pub fn discover_benchmark_dir(root: &Path, name: &str) -> PathBuf { root.join(name) }
