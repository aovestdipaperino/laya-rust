//! End-to-end checks. Skipped unless the checkpoint is present at `models/laya-base`
//! (or wherever `LAYA_MODEL_DIR` points).

use laya::{Agent, Options, Question};
use serde_json::json;
use std::path::PathBuf;

fn agent() -> Option<Agent> {
    let dir = PathBuf::from(
        std::env::var("LAYA_MODEL_DIR").unwrap_or_else(|_| "models/laya-base".to_string()),
    );
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: no checkpoint at {}", dir.display());
        return None;
    }
    Some(Agent::from_dir(dir, Options::default()).expect("loading the checkpoint"))
}

#[test]
fn answers_are_well_formed_and_batching_does_not_change_them() {
    let Some(agent) = agent() else { return };
    let state = json!("My card was charged twice for order #4417. Please refund the duplicate.");

    let questions: Vec<(String, Question)> = vec![
        (
            "department".into(),
            Question::choice("Which team owns this?", ["billing", "technical", "sales"]),
        ),
        (
            "urgency".into(),
            Question::score("How urgent?", ["not urgent", "low", "normal", "high"]),
        ),
        (
            "needs_human".into(),
            Question::noul("A human must handle this."),
        ),
    ];

    let batched = agent.system_one(&state, &questions).unwrap();
    assert_eq!(batched.answers.len(), 3);
    assert!(batched.usage.input_tokens > 0);

    // Each answer must be tagged with, and carry the payload of, its own question type.
    for (id, q) in &questions {
        let a = batched.answers.get(id).expect("missing answer");
        assert_eq!(
            a["type"].as_str().unwrap(),
            q.qtype.as_str(),
            "question {id}"
        );
        let payload = q.qtype.as_str();
        assert!(
            a.get(payload).is_some(),
            "question {id} has no {payload} field"
        );
        assert!(
            a["rl_agent"]["act_probability"].is_number(),
            "question {id}"
        );
    }

    // The three questions padded to a common length in the batch above; run alone, the
    // padding and marker layout differ, so this pins the masking down.
    for (id, q) in &questions {
        let single = agent
            .system_one(&state, &vec![(id.clone(), q.clone())])
            .unwrap();
        let a = single.answers.get(id).unwrap();
        let b = batched.answers.get(id).unwrap();
        assert_eq!(
            a, b,
            "question {id}: batched and single-question answers differ"
        );
    }
}

#[test]
fn choice_probabilities_sum_to_one_and_pick_the_argmax() {
    let Some(agent) = agent() else { return };
    let state = json!("The payment gateway returned a 500 and no orders can be placed.");
    let q = Question::choice("Which team owns this?", ["billing", "technical", "sales"]);
    let out = agent.system_one(&state, &vec![("dept".into(), q)]).unwrap();

    let a = out.answers.get("dept").unwrap();
    let probs = a["probabilities"].as_object().unwrap();
    assert_eq!(probs.len(), 3);
    let sum: f64 = probs.values().map(|v| v.as_f64().unwrap()).sum();
    assert!((sum - 1.0).abs() < 1e-3, "probabilities sum to {sum}");

    let best = probs
        .iter()
        .max_by(|x, y| {
            x.1.as_f64()
                .unwrap()
                .partial_cmp(&y.1.as_f64().unwrap())
                .unwrap()
        })
        .unwrap()
        .0;
    assert_eq!(a["choice"].as_str().unwrap(), best);

    let conf = a["confidence"].as_f64().unwrap();
    assert!(
        (0.0..=1.0).contains(&conf),
        "confidence {conf} out of range"
    );
}

#[test]
fn a_score_answer_stays_within_its_level_range() {
    let Some(agent) = agent() else { return };
    let levels = ["not urgent", "low", "normal", "high", "critical"];
    let out = agent
        .system_one(
            &json!("Production is down for every customer."),
            &vec![("urgency".into(), Question::score("How urgent?", levels))],
        )
        .unwrap();
    let s = out.answers["urgency"]["score"].as_f64().unwrap();
    assert!((0.0..=4.0).contains(&s), "score {s} outside 0..=4");
    assert_eq!(
        out.answers["urgency"]["legend"].as_object().unwrap().len(),
        5
    );
}

#[test]
fn too_many_options_fail_loudly_rather_than_silently_truncating() {
    let Some(agent) = agent() else { return };
    // head_max_len is 192 tokens; 200 options cannot all keep a marker.
    let many: Vec<String> = (0..200).map(|i| format!("category number {i}")).collect();
    let err = agent
        .system_one(
            &json!("anything"),
            &vec![("q".into(), Question::choice("pick one", many))],
        )
        .unwrap_err();
    let msg = format!("{err:#}");
    assert!(msg.contains("head_max_len"), "unexpected error: {msg}");
}
