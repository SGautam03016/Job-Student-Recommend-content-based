import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Data
students_df = pd.read_csv("Data\\students.csv")
jobs_df = pd.read_csv("Data\\jobs.csv")

# Experience Mapping
experience_mapping = {
    "Fresher": 0,
    "1 year": 1, 
    "2 years": 2, 
    "3 years": 3, 
    "4 years": 4, 
    "5+ years": 5, 
}

# Apply Mapping (Standardizing Case & Spaces)
students_df["Experience"] = students_df["Experience"].str.strip().map(experience_mapping)
jobs_df["Experience_Level"] = jobs_df["Experience_Level"].str.strip().map(experience_mapping)

# Convert text-based skills into numerical representation using TF-IDF
vectorizer = TfidfVectorizer()
std_skills_matrix = vectorizer.fit_transform(students_df['Skills'])
job_skills_matrix = vectorizer.transform(jobs_df['Required_Skills'])  # Use transform for consistent features

# Compute Cosine Similarity
similarity_matrix = cosine_similarity(std_skills_matrix, job_skills_matrix)

# Function to recommend students for a given job ID
def recommend_students(job_id, top_n=5):
    # Find job index
    job_idx = jobs_df[jobs_df['Job_ID'] == job_id].index
    if len(job_idx) == 0:
        return pd.DataFrame()  # Return empty DataFrame if job ID is invalid
    job_idx = job_idx[0]

    # Get similarity scores for this job with all students
    similarity_scores = similarity_matrix[:, job_idx]

    # Add similarity scores to students DataFrame
    students_df["Match_Score"] = similarity_scores

    # Filter eligible students based on experience criteria
    job_experience_required = jobs_df.loc[job_idx, "Experience_Level"]
    eligible_students = students_df[students_df["Experience"] >= job_experience_required]

    # Sort students by best match score and return top N
    return eligible_students.sort_values(by="Match_Score", ascending=False).head(top_n)

# Test Recommendation for an Even Job ID (e.g., Job ID 2)
recommended_students = recommend_students(12)
print(recommended_students[["Name", "Skills", "Experience", "Match_Score"]])
