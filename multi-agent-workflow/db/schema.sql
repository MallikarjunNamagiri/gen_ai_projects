CREATE TABLE IF NOT EXISTS workflow_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    input_filename TEXT NOT NULL,
    final_status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_trace (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER REFERENCES workflow_runs(id),
    agent_name TEXT NOT NULL,
    input_summary TEXT,
    output_summary TEXT,
    duration_ms INTEGER,
    step_order INTEGER
);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER REFERENCES workflow_runs(id),
    routing_label TEXT NOT NULL,
    reason TEXT,
    resolved_by TEXT NOT NULL
);