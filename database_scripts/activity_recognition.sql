 -- CREATE DATABASE activity_recognition_test;

CREATE TABLE activities (
    id SERIAL PRIMARY KEY,
    video_name TEXT,
    timestamp TIMESTAMP NOT NULL,
    action_1 TEXT,
    score_1 FLOAT,
    action_2 TEXT,
    score_2 FLOAT,
    action_3 TEXT,
    score_3 FLOAT,
    action_4 TEXT,
    score_4 FLOAT,
    action_5 TEXT,
    score_5 FLOAT,
    sit_score FLOAT,
    stand_score FLOAT,
    lie_score FLOAT,
    walk_score FLOAT
);
