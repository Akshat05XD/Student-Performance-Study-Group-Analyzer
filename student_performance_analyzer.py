"""
========================================================
  MINOR PROJECT
  Title : Student Performance & Study Group Analyzer
  Domain: Python + Statistics + Probability + Linear Algebra + Basic ML
  Author: Akshat Rawat
========================================================

Idea of the project (in simple words):
---------------------------------------
We have marks of 30 students in Math, Science and English, along with
their attendance %. Using this data we:
    1) Clean the data (some marks are missing on purpose)
    2) Find class average, median, std-deviation (Statistics)
    3) Find probability of a student scoring above 80% (Probability)
    4) Treat every student's marks as a "vector" and use Cosine
       Similarity (Linear Algebra / Dot Product) to pair up students
       with similar performance as "Study Buddies"
    5) BONUS: Train a very small ML model (Logistic Regression) that
       predicts whether a student will PASS or FAIL just by looking
       at Attendance % and Science/English marks, and then check how
       good the model is using Accuracy and F1-Score.

This is a beginner level project, so the code is kept simple and every
important line is commented like I would explain it to my teacher.
"""

import numpy as np
import pandas as pd
import random
import matplotlib
matplotlib.use("Agg")            
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# PHASE 0 : GENERATE THE DATASET

def generate_dataset(path="student_data.csv"):
    np.random.seed(42)         
    random.seed(42)             

    first_names = ["Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Ayaan", "Krishna",
                   "Ishaan", "Shaurya", "Ananya", "Aadhya", "Diya", "Saanvi", "Priya", "Neha",
                   "Riya", "Kriti", "Pooja", "Kavya", "Rahul", "Rohan", "Amit", "Karan",
                   "Vikram", "Sneha", "Nidhi", "Tanvi", "Rashi", "Meera"]

    data = []
    for name in first_names:
        attendance = np.random.randint(60, 101)          # attendance between 60% - 100%
        base_score = attendance * 0.8                      

        # add some random "noise" so marks are not exactly predictable
        # (np.random.normal(mean, spread) -> gives realistic looking marks)
        math_grade = min(100, max(0, int(np.random.normal(base_score, 10))))
        science_grade = min(100, max(0, int(np.random.normal(base_score, 8))))
        english_grade = min(100, max(0, int(np.random.normal(base_score + 5, 7))))

        # 5% chance a student's Math marks were not entered (missing value)
        if random.random() < 0.05:
            math_grade = np.nan

        data.append([name, math_grade, science_grade, english_grade, attendance])

    df = pd.DataFrame(data, columns=["Student_Name", "Math_Score", "Science_Score",
                                      "English_Score", "Attendance_Pct"])
    df.to_csv(path, index=False)
    return df


# PHASE 1 : DATA SETUP & CLEANING
def load_and_clean_data(path="student_data.csv"):
    df = pd.read_csv(path)

    # Check how many missing values exist before cleaning
    missing_before = df.isnull().sum().sum()

    # Fill missing marks with the average (mean) marks of that subject.
    # This is a common beginner-friendly way to "handle missing data"
    # instead of just deleting the row.
    df["Math_Score"] = df["Math_Score"].fillna(df["Math_Score"].mean())

    print(f"[Phase 1] Missing values found : {missing_before}")
    print("[Phase 1] Missing values filled using column mean.\n")
    return df


def plot_class_averages(df, out_path="class_average_barchart.png"):
    # Simple bar chart -> average score of the whole class per subject
    subjects = ["Math_Score", "Science_Score", "English_Score"]
    averages = [df[s].mean() for s in subjects]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(["Math", "Science", "English"], averages,
                    color=["#4C72B0", "#55A868", "#C44E52"])
    plt.title("Class Average Score per Subject")
    plt.ylabel("Average Marks")
    plt.ylim(0, 100)
    # Write the exact number on top of every bar (looks nicer in viva/demo)
    for bar, avg in zip(bars, averages):
        plt.text(bar.get_x() + bar.get_width() / 2, avg + 1, f"{avg:.1f}", ha="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Phase 1] Bar chart saved -> {out_path}\n")


# PHASE 2 : STATISTICAL ANALYSIS
def statistical_summary(df):
    subjects = ["Math_Score", "Science_Score", "English_Score"]
    print("[Phase 2] Statistical Summary")
    print("-" * 50)
    summary = {}
    for s in subjects:
        mean_val = df[s].mean()
        median_val = df[s].median()
        std_val = df[s].std()          # standard deviation = how spread out the marks are
        summary[s] = {"mean": mean_val, "median": median_val, "std": std_val}
        print(f"{s:15s} -> Mean: {mean_val:6.2f} | Median: {median_val:6.2f} | Std Dev: {std_val:6.2f}")

    # correlation between attendance and overall (average) score
    df["Overall_Avg"] = df[subjects].mean(axis=1)
    correlation = df["Attendance_Pct"].corr(df["Overall_Avg"])
    print(f"\nCorrelation between Attendance and Overall Score : {correlation:.3f}")
    print("(closer to +1 means: more attendance -> more marks)\n")
    return summary, correlation


# PHASE 3 : PROBABILITY
def probability_analysis(df):
    total_students = len(df)

    # Empirical probability = (favourable outcomes) / (total outcomes)
    # 1) Probability that a randomly picked student scores above 80% overall
    above_80 = df[df["Overall_Avg"] > 80].shape[0]
    prob_above_80 = above_80 / total_students

    # 2) Probability that a randomly picked student PASSES (we take 40% as pass mark)
    passed = df[df["Overall_Avg"] >= 40].shape[0]
    prob_pass = passed / total_students

    print("[Phase 3] Probability Analysis")
    print("-" * 50)
    print(f"P(score > 80%)      = {above_80}/{total_students} = {prob_above_80:.3f}")
    print(f"P(student passes)   = {passed}/{total_students} = {prob_pass:.3f}\n")
    return prob_above_80, prob_pass


# PHASE 4 : STUDY BUDDY MATCHING (LINEAR ALGEBRA -> COSINE SIMILARITY)
def cosine_similarity_manual(vec_a, vec_b):
    # Cosine Similarity = (A . B) / (||A|| * ||B||)
    # A . B        -> dot product of the two vectors
    # ||A||, ||B|| -> magnitude (length) of each vector
    dot_product = np.dot(vec_a, vec_b)
    magnitude_a = np.linalg.norm(vec_a)
    magnitude_b = np.linalg.norm(vec_b)
    if magnitude_a == 0 or magnitude_b == 0:
        return 0
    return dot_product / (magnitude_a * magnitude_b)


def find_study_buddies(df):
    subjects = ["Math_Score", "Science_Score", "English_Score"]
    names = df["Student_Name"].values
    # every row (student) becomes a 3-dimensional vector: [Math, Science, English]
    vectors = df[subjects].values

    n = len(names)
    best_match_idx = np.zeros(n, dtype=int)
    best_match_score = np.zeros(n)

    # compare every student with every other student (this is the simple,
    # beginner-friendly way -> O(n^2), fine for small class sizes)
    for i in range(n):
        best_score = -1
        best_j = -1
        for j in range(n):
            if i == j:
                continue                       # don't compare a student with themself
            score = cosine_similarity_manual(vectors[i], vectors[j])
            if score > best_score:
                best_score = score
                best_j = j
        best_match_idx[i] = best_j
        best_match_score[i] = best_score

    pairs_df = pd.DataFrame({
        "Student": names,
        "Study_Buddy": names[best_match_idx],
        "Similarity_Score": best_match_score.round(4)
    })

    print("[Phase 4] Study Buddy Pairing (Top 10 shown)")
    print("-" * 50)
    print(pairs_df.head(10).to_string(index=False))
    print()
    return pairs_df


# ---------------------------------------------------------------
# PHASE 5 (BONUS) : ML MODEL - PREDICT DISTINCTION / NOT
# ---------------------------------------------------------------
# NOTE: With this dataset every single student clears the basic 40%
# pass mark (attendance in our data is always >= 60%, so nobody
# actually fails). A "Pass vs Fail" model would therefore have only
# ONE class to learn from, which is meaningless for a classifier.
# So, to make the ML part actually useful, we instead predict a more
# realistic and useful label: "Will this student score a DISTINCTION
# (Overall_Avg >= 75%) or NOT?" This still has both classes present
# and is a genuinely useful prediction for a teacher/mentor.
#
# Features used: Attendance_Pct, Science_Score, English_Score
# (Math_Score / Overall_Avg are excluded on purpose since the label
# is directly derived from them -> using them would be "data leakage",
# i.e. the model would just be cheating instead of learning a pattern).
def train_pass_fail_model(df):
    df = df.copy()
    df["Label_Distinction"] = (df["Overall_Avg"] >= 75).astype(int)   # 1 = Distinction, 0 = Not

    features = ["Attendance_Pct", "Science_Score", "English_Score"]
    X = df[features]
    y = df["Label_Distinction"]

    print(f"[Phase 5 - Bonus] Class balance -> Distinction: {y.sum()}, Not: {len(y) - y.sum()}\n")

    # Since our dataset is small (30 students) we still do a proper
    # train/test split (70% train, 30% test) to test the model fairly
    # on data it has never seen. stratify=y keeps the same class ratio
    # in both train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)                # training the model

    y_pred = model.predict(X_test)             # predicting on unseen test data

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("[Phase 5 - Bonus] Distinction Prediction Model (Logistic Regression)")
    print("-" * 50)
    print(f"Train samples: {len(X_train)}  |  Test samples: {len(X_test)}")
    print(f"Accuracy : {acc:.3f}")
    print(f"F1 Score : {f1:.3f}")
    print("Confusion Matrix:")
    print(cm)
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return acc, f1

# MAIN -> RUN ALL PHASES ONE BY ONE
if __name__ == "__main__":
    print("=" * 60)
    print(" STUDENT PERFORMANCE & STUDY GROUP ANALYZER - MINOR PROJECT")
    print("=" * 60, "\n")

    generate_dataset("student_data.csv")
    df = load_and_clean_data("student_data.csv")
    plot_class_averages(df)
    statistical_summary(df)
    probability_analysis(df)
    find_study_buddies(df)
    train_pass_fail_model(df)

    print("Project run completed successfully. Check student_data.csv and")
    print("class_average_barchart.png in the same folder for outputs.")
