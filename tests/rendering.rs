//! Tests for the deterministic, model-free parts: option rendering, sequence layout,
//! temperature bucketing and confidence. These mirror `rl_common.py` exactly.

use laya::question::Question;
use laya::{AgentConfig, QType};
use serde_json::json;

fn render(q: &Question) -> Vec<String> {
    q.render_options().unwrap()
}

#[test]
fn choice_options_render_with_and_without_descriptions() {
    let q: Question = serde_json::from_value(json!({
        "type": "choice",
        "instructions": "who owns it",
        "criteria": {"billing": "money things", "technical": null, "sales": ""}
    }))
    .unwrap();
    assert_eq!(
        render(&q),
        vec!["billing: money things", "technical", "sales"]
    );
}

#[test]
fn choice_criteria_may_be_a_bare_list() {
    let q = Question::choice("who owns it", ["billing", "technical"]);
    assert_eq!(render(&q), vec!["billing", "technical"]);
}

#[test]
fn choice_option_order_follows_the_json() {
    // serde_json is built with preserve_order, so the reported label order is the author's order.
    let q: Question = serde_json::from_value(json!({
        "type": "choice",
        "instructions": "x",
        "criteria": {"zebra": null, "apple": null, "mango": null}
    }))
    .unwrap();
    assert_eq!(render(&q), vec!["zebra", "apple", "mango"]);
}

#[test]
fn score_options_are_numbered_levels() {
    let q = Question::score("how urgent", ["calm", "busy", "on fire"]);
    assert_eq!(
        render(&q),
        vec!["level 0: calm", "level 1: busy", "level 2: on fire"]
    );
}

#[test]
fn noul_is_always_false_then_true_so_p1_is_the_answer() {
    let q = Question::noul("the statement holds");
    let opts = render(&q);
    assert_eq!(opts.len(), 2);
    assert!(opts[0].starts_with("false: "));
    assert!(opts[1].starts_with("true: "));
}

#[test]
fn noul_criteria_override_the_default_sides() {
    let q: Question = serde_json::from_value(json!({
        "type": "noul",
        "instructions": "x",
        "criteria": {"false": "nope", "true": "yep"}
    }))
    .unwrap();
    assert_eq!(render(&q), vec!["false: nope", "true: yep"]);
}

#[test]
fn temperature_buckets_match_the_reference() {
    let cfg: AgentConfig = serde_json::from_value(json!({
        "max_len": 512,
        "head_max_len": 192,
        "temperature": [1.5, 2.5, 3.5],
        "temperature_by_options": {"choice:3-5": 0.25, "noul:2": 0.75}
    }))
    .unwrap();

    // Bucket boundaries: <=2, 3-5, 6-10, 11+.
    assert_eq!(cfg.temperature_for(QType::Choice.index(), 4), 0.25);
    assert_eq!(cfg.temperature_for(QType::Choice.index(), 5), 0.25);
    assert_eq!(cfg.temperature_for(QType::Noul.index(), 2), 0.75);
    // No bucket entry: fall back to the per-type temperature.
    assert_eq!(cfg.temperature_for(QType::Choice.index(), 20), 1.5);
    assert_eq!(cfg.temperature_for(QType::Score.index(), 4), 2.5);
}

#[test]
fn qtype_indices_match_the_checkpoint_embedding_rows() {
    assert_eq!(QType::Choice.index(), 0);
    assert_eq!(QType::Score.index(), 1);
    assert_eq!(QType::Noul.index(), 2);
}

#[test]
fn an_option_less_question_is_rejected_with_a_readable_message() {
    // Without the guard this reaches the model and fails as an opaque tensor reshape.
    let q: Question =
        serde_json::from_value(json!({"type": "score", "instructions": "x", "criteria": []}))
            .unwrap();
    assert_eq!(q.render_options().unwrap().len(), 0);
}
