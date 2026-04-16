import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset into a pandas DataFrame
df = pd.read_csv("dataset.csv")

# Display the top 5 rows
print("Top 5 rows:")
print(df.head())

# Display the bottom 5 rows
print("\nBottom 5 rows:")
print(df.tail())

# Check data types
print("\ndf.dtypes:")
print(df.dtypes)

# Drop irrelevant columns
df.drop(columns=[
    'Curricular units 1st sem (credited)',
    'Curricular units 2nd sem (credited)',
    'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (without evaluations)',
    'Application order',
    'Nacionality',
    'Inflation rate',
    'GDP'
], inplace=True)

# Rename columns for clarity
df.rename(columns={
    "Daytime/evening attendance":             "Attendance_Schedule",
    "Tuition fees up to date":                "Tuition_Fees_Paid",
    "Educational special needs":              "Special_Needs",
    "Curricular units 1st sem (enrolled)":    "Units_1st_Enrolled",
    "Curricular units 1st sem (evaluations)": "Evals_1st_Sem",
    "Curricular units 1st sem (approved)":    "Units_1st_Approved",
    "Curricular units 1st sem (grade)":       "Grade_1st_Sem",
    "Curricular units 2nd sem (enrolled)":    "Units_2nd_Enrolled",
    "Curricular units 2nd sem (evaluations)": "Evals_2nd_Sem",
    "Curricular units 2nd sem (approved)":    "Units_2nd_Approved",
    "Curricular units 2nd sem (grade)":       "Grade_2nd_Sem",
    "Mother's qualification":                 "Mother_Education",
    "Father's qualification":                 "Father_Education",
    "Mother's occupation":                    "Mother_Occupation",
    "Father's occupation":                    "Father_Occupation",
    "Target":                                 "Student_Status"
}, inplace=True)

# Obtain the shape of the DataFrame
print("\ndf.shape:", df.shape)

# Check for duplicate rows
duplicate_rows_df = df[df.duplicated()]
print("\nNumber of duplicate rows: ", duplicate_rows_df.shape)

# Remove duplicates
df = df.drop_duplicates()

# Count remaining rows after removal
print("Rows after duplicate removal:", len(df))

# Drop missing or null values
df = df.dropna()
print("Rows after dropna():", len(df))

# Boxplots
sns.boxplot(x=df['Grade_1st_Sem'])
plt.title("Boxplot - Grade 1st Semester")
plt.tight_layout()
plt.show()

sns.boxplot(x=df['Grade_2nd_Sem'])
plt.title("Boxplot - Grade 2nd Semester")
plt.tight_layout()
plt.show()

sns.boxplot(x=df['Age at enrollment'])
plt.title("Boxplot - Age at Enrollment")
plt.tight_layout()
plt.show()

# Scatter plots
plt.figure(figsize=(8, 5))
plt.scatter(df['Grade_1st_Sem'], df['Grade_2nd_Sem'], alpha=0.4, color='steelblue', s=15)
plt.xlabel("Grade 1st Semester")
plt.ylabel("Grade 2nd Semester")
plt.title("Scatter Plot: Grade 1st Sem vs Grade 2nd Sem")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(df['Age at enrollment'], df['Grade_1st_Sem'], alpha=0.4, color='darkorange', s=15)
plt.xlabel("Age at Enrollment")
plt.ylabel("Grade 1st Semester")
plt.title("Scatter Plot: Age at Enrollment vs Grade 1st Sem")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(df['Age at enrollment'], df['Grade_2nd_Sem'], alpha=0.4, color='darkorange', s=15)
plt.xlabel("Age at Enrollment")
plt.ylabel("Grade 2nd Semester")
plt.title("Scatter Plot: Age at Enrollment vs Grade 2nd Sem")
plt.tight_layout()
plt.show()

# Histograms
df['Grade_1st_Sem'].hist(bins=30)
plt.title("Histogram - Grade 1st Semester")
plt.xlabel("Grade")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

df['Grade_2nd_Sem'].hist(bins=30)
plt.title("Histogram - Grade 2nd Semester")
plt.xlabel("Grade")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

df['Age at enrollment'].hist(bins=30)
plt.title("Histogram - Age at Enrollment")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Correlation heatmap
numeric_cols = [
    'Grade_1st_Sem', 'Grade_2nd_Sem',
    'Units_1st_Enrolled', 'Units_1st_Approved',
    'Units_2nd_Enrolled', 'Units_2nd_Approved',
    'Age at enrollment', 'Unemployment rate'
]
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
