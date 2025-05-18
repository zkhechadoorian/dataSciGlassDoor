import pandas as pd 

df = pd.read_csv('glassdoor_jobs.csv')

#salary parsing 

# create a new feature 'hourly' boolean 1 if wage listed as hourly 0 otherwise
df['hourly'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
# create a new feature: boolean 1 if employer listed salary 0 if it's an estimate
df['employer_provided'] = df['Salary Estimate'].apply(lambda x: 1 if 'employer provided salary:' in x.lower() else 0)
# remove rows missing a salary estimate
df = df[df['Salary Estimate'] != '-1']
# some salaries have (Glassdoor est.) next to their quantity
# if so, we can isolate only the numeric part
salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
# get rid of non-numeric characters
minus_Kd = salary.apply(lambda x: x.replace('K','').replace('$',''))
# get rid of non-numeric phrases
min_hr = minus_Kd.apply(lambda x: x.lower().replace('per hour','').replace('employer provided salary:',''))

# salries are given in a range, we'll create a min/max/avg
df['min_salary'] = min_hr.apply(lambda x: int(x.split('-')[0]))
df['max_salary'] = min_hr.apply(lambda x: int(x.split('-')[1]))
df['avg_salary'] = (df.min_salary+df.max_salary)/2

#Company name text only
df['company_txt'] = df.apply(lambda x: x['Company Name'] if x['Rating'] <0 else x['Company Name'][:-3], axis = 1)

#state field 
df['job_state'] = df['Location'].apply(lambda x: x.split(',')[1])
df.job_state.value_counts()

df['same_state'] = df.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis = 1)

#age of company 
df['age'] = df.Founded.apply(lambda x: x if x <1 else 2020 - x)

#parsing of job description (python, etc.)

#python
df['python_yn'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
 
#r studio 
df['R_yn'] = df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() or 'r-studio' in x.lower() else 0)
df.R_yn.value_counts()

#spark 
df['spark'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)
df.spark.value_counts()

#aws 
df['aws'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)
df.aws.value_counts()

#excel
df['excel'] = df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)
df.excel.value_counts()

df.columns

df_out = df.drop(['Unnamed: 0'], axis =1)

df_out.to_csv('salary_data_cleaned.csv',index = False)

