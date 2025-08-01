# Course Rating & Review Scoring

This project analyzes user ratings and reviews from an online course platform.  
The goal is to go beyond a basic average score by applying time and user-based weighting, and also ranking user reviews more fairly.

---

## 📁 Dataset

- **File:** `course_reviews.csv`
- **Rows/Columns:** 4323 × 6  
- **Columns:**
  - `Rating`: User rating (1–5)
  - `Timestamp`: Date of the rating
  - `Enrolled`: Type of enrollment
  - `Progress`: % of course watched
  - `Questions Asked`: Number of questions asked
  - `Questions Answered`: Number of questions answered

---

## 🎯 What’s the Goal?

- Calculate a more realistic course rating (not just a simple mean).
- Give more value to recent reviews and active users.
- Sort comments in a way that highlights the most useful ones.

---

## 🔍 Methods Used

### 1. Simple Average Rating  
Basic mean of all ratings.

### 2. Time-Based Weighted Average  
Recent reviews get more weight.  
Older ones affect the score less.  
Used 4 time periods:  
0–30 days, 31–90, 91–180, 180+ days.

### 3. User-Based Weighted Average  
Users who watched more of the course get more weight.  
Weights based on progress levels like 0–10%, 11–45%, etc.

### 4. Final Score  
```python
final = (time_weighted * 0.5) + (progress_weighted * 0.5)
```

---

## 💬 Review Scoring (Wilson Lower Bound)

We tested different ways to sort comments:  
- Up–Down difference  
- Average rating  
- **Wilson Lower Bound Score**: more fair when vote counts are small.

---

## ⚙️ How to Run

```bash
pip install pandas scipy scikit-learn
python rating.py
```

---

## 🧾 Sample Output

```
Average Rating: 4.76
Time-Based Weighted Average: 4.74
User-Based Weighted Average: 4.79
Final Rating (50/50): 4.76
Final Rating (40/60): 4.79

Top reviews sorted by Wilson Lower Bound:
...
```

---

## 👩‍💻 Notes

This project was done as part of a learning process.  
Focus was on combining multiple scoring strategies to reflect course value better.
