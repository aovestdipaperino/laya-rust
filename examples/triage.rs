//! Triage a support ticket in one forward pass.
//!
//! `cargo run --release --example triage`

use anyhow::Result;
use laya::{Agent, Options, Question};
use serde_json::json;

fn main() -> Result<()> {
    let agent = Agent::from_dir("models/laya-base", Options::default())?;

    let ticket = json!({
        "subject": "Charged twice for order #4417",
        "body": "I was billed 79.00 EUR on Tuesday and again on Wednesday for the same order. \
                 Please refund the duplicate charge.",
        "customer_tier": "pro"
    });

    let questions = vec![
        (
            "department".to_string(),
            Question::choice(
                "Which team should own this ticket?",
                ["billing", "technical", "sales"],
            ),
        ),
        (
            "urgency".to_string(),
            Question::score(
                "How urgent is this ticket?",
                ["not urgent", "low", "normal", "high", "critical outage"],
            ),
        ),
        (
            "needs_human".to_string(),
            Question::noul("This ticket requires a human agent rather than an automated reply."),
        ),
    ];

    let start = std::time::Instant::now();
    let response = agent.system_one(&ticket, &questions)?;
    eprintln!("{} questions in {:?}", questions.len(), start.elapsed());

    println!("{}", serde_json::to_string_pretty(&response)?);
    Ok(())
}
