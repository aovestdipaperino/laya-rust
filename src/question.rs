//! Typed questions and their rendering into an encoder sequence.
//!
//! Layout, matching `build_sequence` in `rl_common.py`:
//! `[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]`
//! Each option is preceded by a `[MASK]` marker; the head scores those marker positions.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokenizers::Tokenizer;

/// Per-option text budget, in tokens. Matches the `[:48]` slice in the reference implementation.
const MAX_OPTION_TOKENS: usize = 48;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QType {
    /// Pick one of N named options.
    Choice,
    /// Ordinal level; the answer is the expectation over level indices.
    Score,
    /// Boolean "does this statement hold"; the answer is P(true).
    Noul,
}

impl QType {
    pub fn index(self) -> usize {
        match self {
            QType::Choice => 0,
            QType::Score => 1,
            QType::Noul => 2,
        }
    }

    pub fn from_index(i: usize) -> Self {
        match i {
            0 => QType::Choice,
            1 => QType::Score,
            _ => QType::Noul,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            QType::Choice => "choice",
            QType::Score => "score",
            QType::Noul => "noul",
        }
    }
}

/// A question in the request shape: `{"type", "instructions", "criteria"}`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Question {
    #[serde(rename = "type")]
    pub qtype: QType,
    /// Free text, or any JSON value (serialized compactly, as the reference does).
    pub instructions: Value,
    #[serde(default)]
    pub criteria: Option<Value>,
}

impl Question {
    pub fn choice<I, S>(instructions: &str, options: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let opts: Vec<Value> = options
            .into_iter()
            .map(|o| Value::String(o.into()))
            .collect();
        Self {
            qtype: QType::Choice,
            instructions: Value::String(instructions.to_string()),
            criteria: Some(Value::Array(opts)),
        }
    }

    pub fn score<I, S>(instructions: &str, levels: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let levels: Vec<Value> = levels
            .into_iter()
            .map(|o| Value::String(o.into()))
            .collect();
        Self {
            qtype: QType::Score,
            instructions: Value::String(instructions.to_string()),
            criteria: Some(Value::Array(levels)),
        }
    }

    pub fn noul(instructions: &str) -> Self {
        Self {
            qtype: QType::Noul,
            instructions: Value::String(instructions.to_string()),
            criteria: None,
        }
    }

    pub(crate) fn instructions_text(&self) -> String {
        match &self.instructions {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        }
    }

    /// The labels a `choice` answer is reported against, in option order.
    pub(crate) fn choice_keys(&self) -> Result<Vec<String>> {
        match &self.criteria {
            Some(Value::Array(a)) => a
                .iter()
                .map(|v| match v {
                    Value::String(s) => Ok(s.clone()),
                    other => Ok(other.to_string()),
                })
                .collect(),
            Some(Value::Object(m)) => Ok(m.keys().cloned().collect()),
            _ => bail!("choice question needs `criteria` as a list or an object"),
        }
    }

    /// Option texts in label-index order, as the model sees them.
    pub fn render_options(&self) -> Result<Vec<String>> {
        match self.qtype {
            QType::Choice => match &self.criteria {
                Some(Value::Array(a)) => Ok(a
                    .iter()
                    .map(|v| match v {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    })
                    .collect()),
                Some(Value::Object(m)) => Ok(m
                    .iter()
                    .map(|(k, v)| match v {
                        Value::Null => k.clone(),
                        Value::String(s) if s.is_empty() => k.clone(),
                        Value::String(s) => format!("{k}: {s}"),
                        other => format!("{k}: {other}"),
                    })
                    .collect()),
                _ => bail!("choice question needs `criteria` as a list or an object"),
            },
            QType::Score => match &self.criteria {
                Some(Value::Array(a)) => Ok(a
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        let c = match v {
                            Value::String(s) => s.clone(),
                            other => other.to_string(),
                        };
                        format!("level {i}: {c}")
                    })
                    .collect()),
                _ => bail!("score question needs `criteria` as a list of levels"),
            },
            QType::Noul => {
                let empty = Map::new();
                let m = match &self.criteria {
                    Some(Value::Object(m)) => m,
                    _ => &empty,
                };
                let side = |k: &str, default: &str| -> String {
                    match m.get(k) {
                        Some(Value::String(s)) if !s.is_empty() => s.clone(),
                        _ => default.to_string(),
                    }
                };
                Ok(vec![
                    format!(
                        "false: {}",
                        side("false", "no, the statement does not hold")
                    ),
                    format!("true: {}", side("true", "yes, the statement holds")),
                ])
            }
        }
    }

    /// The legend for a `score` answer: level index -> level description.
    pub(crate) fn score_levels(&self) -> Vec<String> {
        match &self.criteria {
            Some(Value::Array(a)) => a
                .iter()
                .map(|v| match v {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                })
                .collect(),
            _ => Vec::new(),
        }
    }
}

/// The state to decide over: free text, or any JSON document.
pub fn serialize_state(state: &Value) -> String {
    match state {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

pub(crate) struct Encoded {
    pub ids: Vec<u32>,
    pub markers: Vec<usize>,
    /// Index of the `[SEP]` that closes the instructions.
    pub head_sep: usize,
    /// Index of the `[SEP]` that closes the option block.
    pub opts_sep: usize,
    /// Number of option labels whose text was cut to fit `head_max_len`.
    pub labels_clipped: usize,
}

pub(crate) struct SpecialIds {
    pub cls: u32,
    pub sep: u32,
    pub mask: u32,
    pub pad: u32,
    pub mask_text: String,
}

fn encode_plain(tok: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    tok.encode(text, false)
        .map(|e| e.get_ids().to_vec())
        .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))
}

/// Shrink every option evenly to fit the head budget, matching the reference. Returns the number
/// of labels whose text was actually cut (`per` is floored at 4, so short labels survive intact).
fn shrink_options(opt_ids: &mut [Vec<u32>], head_max_len: usize) -> usize {
    let used: usize = opt_ids.iter().map(|o| o.len()).sum();
    if head_max_len as isize - used as isize >= 16 {
        return 0;
    }
    // Too many or too long options: shrink every option evenly so the instructions still fit.
    let per = std::cmp::max(
        4,
        head_max_len.saturating_sub(16) / std::cmp::max(1, opt_ids.len()),
    );
    let mut clipped = 0;
    for o in opt_ids.iter_mut() {
        if o.len() > per {
            o.truncate(per);
            clipped += 1;
        }
    }
    clipped
}

/// Cut a tokenized state to `room` tokens. `keep_end` keeps the tail (latest content); the
/// default keeps the head. Matches `st[:room]` / `st[-room:]` in the reference.
fn truncate_state(st: &mut Vec<u32>, room: usize, keep_end: bool) {
    if keep_end {
        if st.len() > room {
            *st = st.split_off(st.len() - room);
        }
    } else {
        st.truncate(room);
    }
}

/// Build the input sequence and the marker positions for one question.
pub(crate) fn build_sequence(
    tok: &Tokenizer,
    sp: &SpecialIds,
    state: &Value,
    q: &Question,
    max_len: usize,
    head_max_len: usize,
    truncate_left: bool,
) -> Result<Encoded> {
    let opts = q.render_options()?;
    if opts.is_empty() {
        bail!(
            "a {} question needs at least one option in `criteria`",
            q.qtype.as_str()
        );
    }
    // A literal [MASK] in user text would be read as a marker, so neutralize it, as the reference does.
    let scrub = |s: &str| s.replace(&sp.mask_text, " ");

    let ins = scrub(&q.instructions_text());
    let mut head_ids = encode_plain(tok, &format!("{} question: {}", q.qtype.as_str(), ins))?;

    let mut opt_ids: Vec<Vec<u32>> = Vec::with_capacity(opts.len());
    for o in &opts {
        let mut ids = vec![sp.mask];
        let mut body = encode_plain(tok, &format!(" {}", scrub(o)))?;
        body.truncate(MAX_OPTION_TOKENS);
        ids.extend(body);
        opt_ids.push(ids);
    }

    let labels_clipped = shrink_options(&mut opt_ids, head_max_len);
    let used: usize = opt_ids.iter().map(|o| o.len()).sum();
    let opt_budget = head_max_len as isize - used as isize;
    head_ids.truncate(std::cmp::max(8, opt_budget.max(0) as usize));

    let mut ids = Vec::with_capacity(max_len);
    ids.push(sp.cls);
    ids.extend_from_slice(&head_ids);
    let head_sep = ids.len();
    ids.push(sp.sep);

    let mut markers = Vec::with_capacity(opt_ids.len());
    for o in &opt_ids {
        markers.push(ids.len());
        ids.extend_from_slice(o);
    }
    let opts_sep = ids.len();
    ids.push(sp.sep);

    let room = max_len.saturating_sub(ids.len() + 1);
    let mut st = encode_plain(tok, &scrub(&serialize_state(state)))?;
    truncate_state(&mut st, room, truncate_left);
    ids.extend_from_slice(&st);
    ids.push(sp.sep);
    ids.truncate(max_len);

    let markers: Vec<usize> = markers.into_iter().filter(|m| *m < max_len).collect();
    if markers.len() != opts.len() {
        bail!(
            "options do not fit in head_max_len={head_max_len} tokens ({} of {} markers kept)",
            markers.len(),
            opts.len()
        );
    }
    Ok(Encoded {
        ids,
        markers,
        head_sep,
        opts_sep,
        labels_clipped,
    })
}

#[cfg(test)]
mod tests {
    use super::{shrink_options, truncate_state};

    #[test]
    fn shrink_options_is_a_no_op_when_the_head_budget_has_room() {
        let mut opts = vec![vec![0, 1, 2, 3], vec![0, 4, 5]];
        assert_eq!(shrink_options(&mut opts, 64), 0);
        assert_eq!(opts[0].len(), 4);
        assert_eq!(opts[1].len(), 3);
    }

    #[test]
    fn shrink_options_clips_deep_labels_to_the_floor_and_counts_them() {
        let mut opts: Vec<Vec<u32>> = vec![
            vec![0, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            vec![0, 20, 21, 22, 23, 24, 25, 26],
            vec![0, 30],
        ];
        let clipped = shrink_options(&mut opts, 16);
        assert_eq!(clipped, 2);
        assert_eq!(opts[0].len(), 4);
        assert_eq!(opts[1].len(), 4);
        // A label that already fits the floor is left alone.
        assert_eq!(opts[2].len(), 2);
    }

    #[test]
    fn truncate_state_keeps_the_head_by_default() {
        let mut st = vec![1, 2, 3, 4, 5];
        truncate_state(&mut st, 3, false);
        assert_eq!(st, vec![1, 2, 3]);
    }

    #[test]
    fn truncate_state_keeps_the_tail_with_keep_end() {
        let mut st = vec![1, 2, 3, 4, 5];
        truncate_state(&mut st, 3, true);
        assert_eq!(st, vec![3, 4, 5]);
    }

    #[test]
    fn truncate_state_is_a_no_op_when_it_fits() {
        let mut st = vec![1, 2, 3];
        truncate_state(&mut st, 10, true);
        assert_eq!(st, vec![1, 2, 3]);
    }
}
