# ML Terminology (Plain English)

This guide explains a few core terms you’ll see in this repo in simple, practical language.

## Learning Rate
- What it is: A knob that controls how big a step the model takes when it learns from its mistakes. Bigger step = faster movement; smaller step = slower movement.
- Why it matters: Too big and you can overshoot the best solution (training bounces or diverges). Too small and training is painfully slow.
- Mental picture: Walking downhill to the bottom of a valley. A big stride gets you there quickly but risks overshooting. Tiny steps are safe but slow.
- In formulas (conceptually): new_value = old_value − learning_rate × error_direction.

## Weight
- What it is: A number that tells the model how important a feature is and in which direction it affects the prediction.
- Positive weight: As the feature increases, the prediction tends to increase.
- Negative weight: As the feature increases, the prediction tends to decrease.
- Magnitude: Bigger absolute value means the feature has a stronger influence.
- Intuition: Think of weights as volume knobs for each input feature.

## Bias
- What it is: The model’s baseline or default prediction when all inputs are zero. In a line equation, it’s the intercept.
- Why it matters: It lets the model shift up/down (or in higher dimensions, shift the surface) to better fit the data.
- Intuition: If you built a house on a hill, the bias is the “height offset” of the ground before you add anything else.

## Euclidean Distance (and KNN)
- What it is: The straight‑line distance between two points. For two points A and B with features (x1, x2, …, xd):
  d(A, B) = sqrt((x1A − x1B)^2 + (x2A − x2B)^2 + … + (xdA − xdB)^2).
- Plain English: “How far apart are these two examples if you measured with a ruler in feature space?”

### Impact on KNN (K‑Nearest Neighbors)
- Core idea: KNN looks at the K closest training points (by distance) to decide a label (classification) or value (regression).
- Similarity = closeness: Smaller Euclidean distance means more similar; those neighbors vote more strongly on the outcome.
- Feature scaling matters: If one feature has a much larger range (e.g., 0–10,000) than another (e.g., 0–10), it will dominate the distance and skew the neighbors. Standardize/normalize features so distances reflect true similarity.
- Choice of K: Small K can be noisy (sensitive to outliers). Larger K is smoother but may blur important local patterns. Typical starting points: K in the range 3–15, then tune.
- High dimensions: In many features, distances tend to look similar (“curse of dimensionality”), making neighbors less meaningful. Dimensionality reduction or feature selection can help.
- Alternatives: Sometimes Manhattan distance (L1) or other metrics work better depending on your data’s geometry.

## Quick Tips
- Scale features before KNN (e.g., standardization: zero mean, unit variance).
- Start with a modest learning rate (e.g., 0.01) and adjust if loss diverges (too big) or trains too slowly (too small).
- Inspect weights to understand feature influence; large magnitudes deserve attention.
- Tune K with validation data; prefer odd K for binary classification to reduce ties.

---
This document is for learning purposes and complements code in this repository.
