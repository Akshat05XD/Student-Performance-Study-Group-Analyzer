# Minor Project Report
## Student Performance & Study Group Analyzer

**Student Name:** Akshat Rawat  

**Tools:** Python / Data Analytics / Applied Mathematics

---

## 1. Objective
To build a beginner-friendly data analysis tool that studies student academic
records (Math, Science, English marks + Attendance %), and to:
- Clean and summarize the data statistically
- Compute probabilities from the grade distribution
- Pair students into "Study Buddy" groups using Cosine Similarity (Linear
  Algebra)
- As a bonus, train a small Machine Learning model and report its final
  **Accuracy** and **F1-Score**

## 2. Dataset
A simulated dataset of **30 students** was generated (`student_data.csv`)
with columns: `Student_Name`, `Math_Score`, `Science_Score`, `English_Score`,
`Attendance_Pct`. Marks were generated so that they realistically correlate
with attendance, and ~5% of Math marks were left blank on purpose to
practice handling missing data.

## 3. Methodology & Results

### Phase 1 — Data Cleaning
- 4 missing values were found in `Math_Score`.
- Filled using the **column mean** (a simple, standard beginner technique).
- A bar chart of class averages per subject was generated
  (`class_average_barchart.png`).

### Phase 2 — Statistical Analysis
| Subject | Mean | Median | Std Dev |
|---|---|---|---|
| Math | 67.65 | 70.50 | 12.89 |
| Science | 67.40 | 70.50 | 13.67 |
| English | 71.13 | 72.00 | 10.96 |

- **Correlation** between Attendance % and Overall Average score = **0.917**
  → a strong positive correlation, i.e. students who attend more classes
  tend to score higher.

### Phase 3 — Probability
- P(a randomly chosen student scores above 80%) = **0.100** (3 out of 30)
- P(a randomly chosen student passes, pass mark = 40%) = **1.000** (30/30)

### Phase 4 — Study Buddy Matching (Linear Algebra)
Each student's 3 marks (Math, Science, English) were treated as a
3-dimensional **vector**. The **Cosine Similarity** (dot product ÷ product of
vector magnitudes) between every pair of students was computed, and each
student was paired with the classmate whose grade profile was most similar
to their own — used to form well-matched study groups. Sample output:

| Student | Study Buddy | Similarity |
|---|---|---|
| Aarav | Shaurya | 0.9995 |
| Vivaan | Sai | 0.9998 |
| Krishna | Kriti | 1.0000 |

### Phase 5 (Bonus) — Machine Learning Model
A **Logistic Regression** classifier was trained to predict whether a
student will score a **Distinction (Overall Average ≥ 75%)** or not, using
`Attendance_Pct`, `Science_Score`, and `English_Score` as input features.
(Math_Score/Overall_Avg were deliberately excluded since the label is
derived from them — using them would cause data leakage.)

- Data split: 70% train (21 students) / 30% test (9 students), stratified
  so both classes appear in the test set.

**Final Model Performance (on unseen test data):**

| Metric | Score |
|---|---|
| **Accuracy** | **0.778 (77.8%)** |
| **F1-Score** | **0.750** |

Confusion Matrix:
```
                Predicted: Not   Predicted: Distinction
Actual: Not            4                 2
Actual: Distinction    0                 3
```

**Interpretation:** Out of 9 unseen students, the model correctly
classified 7. It correctly identified all 3 actual distinction-holders
(recall = 1.00 for that class) but was a bit cautious and mis-flagged 2
"Not" students as potential distinction-holders (precision = 0.60 for that
class). Given the very small dataset (30 rows total), this is a
reasonable result for a minor/beginner project.

## 4. Note on Dataset Size
This is a simulated dataset of only 30 students, used to demonstrate the
concepts clearly. The accuracy/F1-score would typically become more stable
and reliable with a larger, real dataset (e.g. 200+ students).

## 5. Tools & Libraries Used
`Python 3`, `pandas`, `numpy`, `matplotlib`, `scikit-learn`
(`train_test_split`, `LogisticRegression`, `accuracy_score`, `f1_score`)

## 6. Files Submitted
- `student_performance_analyzer.py` — main project code (commented)
- `student_data.csv` — generated dataset
- `class_average_barchart.png` — output chart
- `README_Project_Report.md` — this report
