# Synthetic Intent Dataset Quality Report

## Overview
- Raw generated: 1800 rows (600/class)
- After MinHash LSH dedup (threshold=0.85): 1709 rows
- Training set saved: 1500 rows (500/class)

## Label Distribution
label
BOTH                500
HEALTH_ADVICE       500
NUTRITION_LOOKUP    500

## Length Stats (tokens/question)
count    1709.0
mean       10.0
std         2.8
min         5.0
25%         8.0
50%        10.0
75%        12.0
max        19.0

## Similarity Analysis
- Avg within-class cosine:  0.378
- Avg between-class cosine: 0.283
- Separation ratio: 1.34x

## Conclusion
Dataset quality is acceptable. Classes are well-separated.
