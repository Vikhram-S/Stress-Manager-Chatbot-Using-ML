import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import streamlit as st
import traceback
from io import StringIO

# Set page configuration first
st.set_page_config(
    page_title="Mental Stress Manager", 
    layout="wide"
)

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'model' not in st.session_state:
    st.session_state.model = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = None

# Load and prepare the dataset from string
data = '''Person ID,Gender,Age,Occupation,Sleep Duration,Quality of Sleep,Physical Activity Level,Stress Level,BMI Category,Blood Pressure,Heart Rate,Daily Steps,Sleep Disorder
1,Male,27,Software Engineer,6.1,6,42,6,Overweight,126/83,77,4200,
2,Male,28,Doctor,6.2,6,60,8,Normal,125/80,75,10000,
3,Male,28,Doctor,6.2,6,60,8,Normal,125/80,75,10000,
4,Male,28,Sales Representative,5.9,4,30,8,Obese,140/90,85,3000,Sleep Apnea
5,Male,28,Sales Representative,5.9,4,30,8,Obese,140/90,85,3000,Sleep Apnea
6,Male,28,Software Engineer,5.9,4,30,8,Obese,140/90,85,3000,Insomnia
7,Male,29,Teacher,6.3,6,40,7,Obese,140/90,82,3500,Insomnia
8,Male,29,Doctor,7.8,7,75,6,Normal,120/80,70,8000,
9,Male,29,Doctor,7.8,7,75,6,Normal,120/80,70,8000,
10,Male,29,Doctor,7.8,7,75,6,Normal,120/80,70,8000,
11,Male,29,Doctor,6.1,6,30,8,Normal,120/80,70,8000,
12,Male,29,Doctor,7.8,7,75,6,Normal,120/80,70,8000,
13,Male,29,Doctor,6.1,6,30,8,Normal,120/80,70,8000,
14,Male,29,Doctor,6.0,6,30,8,Normal,120/80,70,8000,
15,Male,29,Doctor,6.0,6,30,8,Normal,120/80,70,8000,
16,Male,29,Doctor,6.0,6,30,8,Normal,120/80,70,8000,
17,Female,29,Nurse,6.5,5,40,7,Normal Weight,132/87,80,4000,Sleep Apnea
18,Male,29,Doctor,6.0,6,30,8,Normal,120/80,70,8000,Sleep Apnea
19,Female,29,Nurse,6.5,5,40,7,Normal Weight,132/87,80,4000,Insomnia
20,Male,30,Doctor,7.6,7,75,6,Normal,120/80,70,8000,
21,Male,30,Doctor,7.7,7,75,6,Normal,120/80,70,8000,
22,Male,30,Doctor,7.7,7,75,6,Normal,120/80,70,8000,
23,Male,30,Doctor,7.7,7,75,6,Normal,120/80,70,8000,
24,Male,30,Doctor,7.7,7,75,6,Normal,120/80,70,8000,
25,Male,30,Doctor,7.8,7,75,6,Normal,120/80,70,8000,
26,Male,30,Doctor,7.9,7,75,6,Normal,120/80,70,8000,
27,Male,30,Doctor,7.8,7,75,6,Normal,120/80,70,8000,
28,Male,30,Doctor,7.9,7,75,6,Normal,120/80,70,8000,
29,Male,30,Doctor,7.9,7,75,6,Normal,120/80,70,8000,
30,Male,30,Doctor,7.9,7,75,6,Normal,120/80,70,8000,
31,Female,30,Nurse,6.4,5,35,7,Normal Weight,130/86,78,4100,Sleep Apnea
32,Female,30,Nurse,6.4,5,35,7,Normal Weight,130/86,78,4100,Insomnia
33,Female,31,Nurse,7.9,8,75,4,Normal Weight,117/76,69,6800,
34,Male,31,Doctor,6.1,6,30,8,Normal,125/80,72,5000,
35,Male,31,Doctor,7.7,7,75,6,Normal,120/80,70,8000,
36,Male,31,Doctor,6.1,6,30,8,Normal,125/80,72,5000,
37,Male,31,Doctor,6.1,6,30,8,Normal,125/80,72,5000,
38,Male,31,Doctor,7.6,7,75,6,Normal,120/80,70,8000,
39,Male,31,Doctor,7.6,7,75,6,Normal,120/80,70,8000,
40,Male,31,Doctor,7.6,7,75,6,Normal,120/80,70,8000,
41,Male,31,Doctor,7.7,7,75,6,Normal,120/80,70,8000,
42,Male,31,Doctor,7.7,7,75,6,Normal,120/80,70,8000,
43,Male,31,Doctor,7.7,7,75,6,Normal,120/80,70,8000,
44,Male,31,Doctor,7.8,7,75,6,Normal,120/80,70,8000,
45,Male,31,Doctor,7.7,7,75,6,Normal,120/80,70,8000,
46,Male,31,Doctor,7.8,7,75,6,Normal,120/80,70,8000,
47,Male,31,Doctor,7.7,7,75,6,Normal,120/80,70,8000,
48,Male,31,Doctor,7.8,7,75,6,Normal,120/80,70,8000,
49,Male,31,Doctor,7.7,7,75,6,Normal,120/80,70,8000,
50,Male,31,Doctor,7.7,7,75,6,Normal,120/80,70,8000,Sleep Apnea
51,Male,32,Engineer,7.5,8,45,3,Normal,120/80,70,8000,
52,Male,32,Engineer,7.5,8,45,3,Normal,120/80,70,8000,
53,Male,32,Doctor,6.0,6,30,8,Normal,125/80,72,5000,
54,Male,32,Doctor,7.6,7,75,6,Normal,120/80,70,8000,
55,Male,32,Doctor,6.0,6,30,8,Normal,125/80,72,5000,
56,Male,32,Doctor,6.0,6,30,8,Normal,125/80,72,5000,
57,Male,32,Doctor,7.7,7,75,6,Normal,120/80,70,8000,
58,Male,32,Doctor,6.0,6,30,8,Normal,125/80,72,5000,
59,Male,32,Doctor,6.0,6,30,8,Normal,125/80,72,5000,
60,Male,32,Doctor,7.7,7,75,6,Normal,120/80,70,8000,
61,Male,32,Doctor,6.0,6,30,8,Normal,125/80,72,5000,
62,Male,32,Doctor,6.0,6,30,8,Normal,125/80,72,5000,
63,Male,32,Doctor,6.2,6,30,8,Normal,125/80,72,5000,
64,Male,32,Doctor,6.2,6,30,8,Normal,125/80,72,5000,
65,Male,32,Doctor,6.2,6,30,8,Normal,125/80,72,5000,
66,Male,32,Doctor,6.2,6,30,8,Normal,125/80,72,5000,
67,Male,32,Accountant,7.2,8,50,6,Normal Weight,118/76,68,7000,
68,Male,33,Doctor,6.0,6,30,8,Normal,125/80,72,5000,Insomnia
69,Female,33,Scientist,6.2,6,50,6,Overweight,128/85,76,5500,
70,Female,33,Scientist,6.2,6,50,6,Overweight,128/85,76,5500,
71,Male,33,Doctor,6.1,6,30,8,Normal,125/80,72,5000,
72,Male,33,Doctor,6.1,6,30,8,Normal,125/80,72,5000,
73,Male,33,Doctor,6.1,6,30,8,Normal,125/80,72,5000,
74,Male,33,Doctor,6.1,6,30,8,Normal,125/80,72,5000,
75,Male,33,Doctor,6.0,6,30,8,Normal,125/80,72,5000,
76,Male,33,Doctor,6.0,6,30,8,Normal,125/80,72,5000,
77,Male,33,Doctor,6.0,6,30,8,Normal,125/80,72,5000,
78,Male,33,Doctor,6.0,6,30,8,Normal,125/80,72,5000,
79,Male,33,Doctor,6.0,6,30,8,Normal,125/80,72,5000,
80,Male,33,Doctor,6.0,6,30,8,Normal,125/80,72,5000,
81,Female,34,Scientist,5.8,4,32,8,Overweight,131/86,81,5200,Sleep Apnea
82,Female,34,Scientist,5.8,4,32,8,Overweight,131/86,81,5200,Sleep Apnea
83,Male,35,Teacher,6.7,7,40,5,Overweight,128/84,70,5600,
84,Male,35,Teacher,6.7,7,40,5,Overweight,128/84,70,5600,
85,Male,35,Software Engineer,7.5,8,60,5,Normal Weight,120/80,70,8000,
86,Female,35,Accountant,7.2,8,60,4,Normal,115/75,68,7000,
87,Male,35,Engineer,7.2,8,60,4,Normal,125/80,65,5000,
88,Male,35,Engineer,7.2,8,60,4,Normal,125/80,65,5000,
89,Male,35,Engineer,7.3,8,60,4,Normal,125/80,65,5000,
90,Male,35,Engineer,7.3,8,60,4,Normal,125/80,65,5000,
91,Male,35,Engineer,7.3,8,60,4,Normal,125/80,65,5000,
92,Male,35,Engineer,7.3,8,60,4,Normal,125/80,65,5000,
93,Male,35,Software Engineer,7.5,8,60,5,Normal Weight,120/80,70,8000,
94,Male,35,Lawyer,7.4,7,60,5,Obese,135/88,84,3300,Sleep Apnea
95,Female,36,Accountant,7.2,8,60,4,Normal,115/75,68,7000,Insomnia
96,Female,36,Accountant,7.1,8,60,4,Normal,115/75,68,7000,
97,Female,36,Accountant,7.2,8,60,4,Normal,115/75,68,7000,
98,Female,36,Accountant,7.1,8,60,4,Normal,115/75,68,7000,
99,Female,36,Teacher,7.1,8,60,4,Normal,115/75,68,7000,
100,Female,36,Teacher,7.1,8,60,4,Normal,115/75,68,7000,
101,Female,36,Teacher,7.2,8,60,4,Normal,115/75,68,7000,
102,Female,36,Teacher,7.2,8,60,4,Normal,115/75,68,7000,
103,Female,36,Teacher,7.2,8,60,4,Normal,115/75,68,7000,
104,Male,36,Teacher,6.6,5,35,7,Overweight,129/84,74,4800,Sleep Apnea
105,Female,36,Teacher,7.2,8,60,4,Normal,115/75,68,7000,Sleep Apnea
106,Male,36,Teacher,6.6,5,35,7,Overweight,129/84,74,4800,Insomnia
107,Female,37,Nurse,6.1,6,42,6,Overweight,126/83,77,4200,
108,Male,37,Engineer,7.8,8,70,4,Normal Weight,120/80,68,7000,
109,Male,37,Engineer,7.8,8,70,4,Normal Weight,120/80,68,7000,
110,Male,37,Lawyer,7.4,8,60,5,Normal,130/85,68,8000,
111,Female,37,Accountant,7.2,8,60,4,Normal,115/75,68,7000,
112,Male,37,Lawyer,7.4,8,60,5,Normal,130/85,68,8000,
113,Female,37,Accountant,7.2,8,60,4,Normal,115/75,68,7000,
114,Male,37,Lawyer,7.4,8,60,5,Normal,130/85,68,8000,
115,Female,37,Accountant,7.2,8,60,4,Normal,115/75,68,7000,
116,Female,37,Accountant,7.2,8,60,4,Normal,115/75,68,7000,
117,Female,37,Accountant,7.2,8,60,4,Normal,115/75,68,7000,
118,Female,37,Accountant,7.2,8,60,4,Normal,115/75,68,7000,
119,Female,37,Accountant,7.2,8,60,4,Normal,115/75,68,7000,
120,Female,37,Accountant,7.2,8,60,4,Normal,115/75,68,7000,
121,Female,37,Accountant,7.2,8,60,4,Normal,115/75,68,7000,
122,Female,37,Accountant,7.2,8,60,4,Normal,115/75,68,7000,
123,Female,37,Accountant,7.2,8,60,4,Normal,115/75,68,7000,
124,Female,37,Accountant,7.2,8,60,4,Normal,115/75,68,7000,
125,Female,37,Accountant,7.2,8,60,4,Normal,115/75,68,7000,
126,Female,37,Nurse,7.5,8,60,4,Normal Weight,120/80,70,8000,
127,Male,38,Lawyer,7.3,8,60,5,Normal,130/85,68,8000,
128,Female,38,Accountant,7.1,8,60,4,Normal,115/75,68,7000,
129,Male,38,Lawyer,7.3,8,60,5,Normal,130/85,68,8000,
130,Male,38,Lawyer,7.3,8,60,5,Normal,130/85,68,8000,
131,Female,38,Accountant,7.1,8,60,4,Normal,115/75,68,7000,
132,Male,38,Lawyer,7.3,8,60,5,Normal,130/85,68,8000,
133,Male,38,Lawyer,7.3,8,60,5,Normal,130/85,68,8000,
134,Female,38,Accountant,7.1,8,60,4,Normal,115/75,68,7000,
135,Male,38,Lawyer,7.3,8,60,5,Normal,130/85,68,8000,
136,Male,38,Lawyer,7.3,8,60,5,Normal,130/85,68,8000,
137,Female,38,Accountant,7.1,8,60,4,Normal,115/75,68,7000,
138,Male,38,Lawyer,7.1,8,60,5,Normal,130/85,68,8000,
139,Female,38,Accountant,7.1,8,60,4,Normal,115/75,68,7000,
140,Male,38,Lawyer,7.1,8,60,5,Normal,130/85,68,8000,
141,Female,38,Accountant,7.1,8,60,4,Normal,115/75,68,7000,
142,Male,38,Lawyer,7.1,8,60,5,Normal,130/85,68,8000,
143,Female,38,Accountant,7.1,8,60,4,Normal,115/75,68,7000,
144,Female,38,Accountant,7.1,8,60,4,Normal,115/75,68,7000,
145,Male,38,Lawyer,7.1,8,60,5,Normal,130/85,68,8000,Sleep Apnea
146,Female,38,Lawyer,7.4,7,60,5,Obese,135/88,84,3300,Sleep Apnea
147,Male,39,Lawyer,7.2,8,60,5,Normal,130/85,68,8000,Insomnia
148,Male,39,Engineer,6.5,5,40,7,Overweight,132/87,80,4000,Insomnia
149,Female,39,Lawyer,6.9,7,50,6,Normal Weight,128/85,75,5500,
150,Female,39,Accountant,8.0,9,80,3,Normal Weight,115/78,67,7500,
151,Female,39,Accountant,8.0,9,80,3,Normal Weight,115/78,67,7500,
152,Male,39,Lawyer,7.2,8,60,5,Normal,130/85,68,8000,
153,Male,39,Lawyer,7.2,8,60,5,Normal,130/85,68,8000,
154,Male,39,Lawyer,7.2,8,60,5,Normal,130/85,68,8000,
155,Male,39,Lawyer,7.2,8,60,5,Normal,130/85,68,8000,
156,Male,39,Lawyer,7.2,8,60,5,Normal,130/85,68,8000,
157,Male,39,Lawyer,7.2,8,60,5,Normal,130/85,68,8000,
158,Male,39,Lawyer,7.2,8,60,5,Normal,130/85,68,8000,
159,Male,39,Lawyer,7.2,8,60,5,Normal,130/85,68,8000,
160,Male,39,Lawyer,7.2,8,60,5,Normal,130/85,68,8000,
161,Male,39,Lawyer,7.2,8,60,5,Normal,130/85,68,8000,
162,Female,40,Accountant,7.2,8,55,6,Normal Weight,119/77,73,7300,
163,Female,40,Accountant,7.2,8,55,6,Normal Weight,119/77,73,7300,
164,Male,40,Lawyer,7.9,8,90,5,Normal,130/85,68,8000,
165,Male,40,Lawyer,7.9,8,90,5,Normal,130/85,68,8000,
166,Male,41,Lawyer,7.6,8,90,5,Normal,130/85,70,8000,Insomnia
167,Male,41,Engineer,7.3,8,70,6,Normal Weight,121/79,72,6200,
168,Male,41,Lawyer,7.1,7,55,6,Overweight,125/82,72,6000,
169,Male,41,Lawyer,7.1,7,55,6,Overweight,125/82,72,6000,
170,Male,41,Lawyer,7.7,8,90,5,Normal,130/85,70,8000,
171,Male,41,Lawyer,7.7,8,90,5,Normal,130/85,70,8000,
172,Male,41,Lawyer,7.7,8,90,5,Normal,130/85,70,8000,
173,Male,41,Lawyer,7.7,8,90,5,Normal,130/85,70,8000,
174,Male,41,Lawyer,7.7,8,90,5,Normal,130/85,70,8000,
175,Male,41,Lawyer,7.6,8,90,5,Normal,130/85,70,8000,
176,Male,41,Lawyer,7.6,8,90,5,Normal,130/85,70,8000,
177,Male,41,Lawyer,7.6,8,90,5,Normal,130/85,70,8000,
178,Male,42,Salesperson,6.5,6,45,7,Overweight,130/85,72,6000,Insomnia
179,Male,42,Lawyer,7.8,8,90,5,Normal,130/85,70,8000,
180,Male,42,Lawyer,7.8,8,90,5,Normal,130/85,70,8000,
181,Male,42,Lawyer,7.8,8,90,5,Normal,130/85,70,8000,
182,Male,42,Lawyer,7.8,8,90,5,Normal,130/85,70,8000,
183,Male,42,Lawyer,7.8,8,90,5,Normal,130/85,70,8000,
184,Male,42,Lawyer,7.8,8,90,5,Normal,130/85,70,8000,
185,Female,42,Teacher,6.8,6,45,7,Overweight,130/85,78,5000,Sleep Apnea
186,Female,42,Teacher,6.8,6,45,7,Overweight,130/85,78,5000,Sleep Apnea
187,Female,43,Teacher,6.7,7,45,4,Overweight,135/90,65,6000,Insomnia
188,Male,43,Salesperson,6.3,6,45,7,Overweight,130/85,72,6000,Insomnia
189,Female,43,Teacher,6.7,7,45,4,Overweight,135/90,65,6000,Insomnia
190,Male,43,Salesperson,6.5,6,45,7,Overweight,130/85,72,6000,Insomnia
191,Female,43,Teacher,6.7,7,45,4,Overweight,135/90,65,6000,Insomnia
192,Male,43,Salesperson,6.4,6,45,7,Overweight,130/85,72,6000,Insomnia
193,Male,43,Salesperson,6.5,6,45,7,Overweight,130/85,72,6000,Insomnia
194,Male,43,Salesperson,6.5,6,45,7,Overweight,130/85,72,6000,Insomnia
195,Male,43,Salesperson,6.5,6,45,7,Overweight,130/85,72,6000,Insomnia
196,Male,43,Salesperson,6.5,6,45,7,Overweight,130/85,72,6000,Insomnia
197,Male,43,Salesperson,6.5,6,45,7,Overweight,130/85,72,6000,Insomnia
198,Male,43,Salesperson,6.5,6,45,7,Overweight,130/85,72,6000,Insomnia
199,Male,43,Salesperson,6.5,6,45,7,Overweight,130/85,72,6000,Insomnia
200,Male,43,Salesperson,6.5,6,45,7,Overweight,130/85,72,6000,Insomnia
201,Male,43,Salesperson,6.5,6,45,7,Overweight,130/85,72,6000,Insomnia
202,Male,43,Engineer,7.8,8,90,5,Normal,130/85,70,8000,Insomnia
203,Male,43,Engineer,7.8,8,90,5,Normal,130/85,70,8000,Insomnia
204,Male,43,Engineer,6.9,6,47,7,Normal Weight,117/76,69,6800,
205,Male,43,Engineer,7.6,8,75,4,Overweight,122/80,68,6800,
206,Male,43,Engineer,7.7,8,90,5,Normal,130/85,70,8000,
207,Male,43,Engineer,7.7,8,90,5,Normal,130/85,70,8000,
208,Male,43,Engineer,7.7,8,90,5,Normal,130/85,70,8000,
209,Male,43,Engineer,7.7,8,90,5,Normal,130/85,70,8000,
210,Male,43,Engineer,7.8,8,90,5,Normal,130/85,70,8000,
211,Male,43,Engineer,7.7,8,90,5,Normal,130/85,70,8000,
212,Male,43,Engineer,7.8,8,90,5,Normal,130/85,70,8000,
213,Male,43,Engineer,7.8,8,90,5,Normal,130/85,70,8000,
214,Male,43,Engineer,7.8,8,90,5,Normal,130/85,70,8000,
215,Male,43,Engineer,7.8,8,90,5,Normal,130/85,70,8000,
216,Male,43,Engineer,7.8,8,90,5,Normal,130/85,70,8000,
217,Male,43,Engineer,7.8,8,90,5,Normal,130/85,70,8000,
218,Male,43,Engineer,7.8,8,90,5,Normal,130/85,70,8000,
219,Male,43,Engineer,7.8,8,90,5,Normal,130/85,70,8000,Sleep Apnea
220,Male,43,Salesperson,6.5,6,45,7,Overweight,130/85,72,6000,Sleep Apnea
221,Female,44,Teacher,6.6,7,45,4,Overweight,135/90,65,6000,Insomnia
222,Male,44,Salesperson,6.4,6,45,7,Overweight,130/85,72,6000,Insomnia
223,Male,44,Salesperson,6.3,6,45,7,Overweight,130/85,72,6000,Insomnia
224,Male,44,Salesperson,6.4,6,45,7,Overweight,130/85,72,6000,Insomnia
225,Female,44,Teacher,6.6,7,45,4,Overweight,135/90,65,6000,Insomnia
226,Male,44,Salesperson,6.3,6,45,7,Overweight,130/85,72,6000,Insomnia
227,Female,44,Teacher,6.6,7,45,4,Overweight,135/90,65,6000,Insomnia
228,Male,44,Salesperson,6.3,6,45,7,Overweight,130/85,72,6000,Insomnia
229,Female,44,Teacher,6.6,7,45,4,Overweight,135/90,65,6000,Insomnia
230,Male,44,Salesperson,6.3,6,45,7,Overweight,130/85,72,6000,Insomnia
231,Female,44,Teacher,6.6,7,45,4,Overweight,135/90,65,6000,Insomnia
232,Male,44,Salesperson,6.3,6,45,7,Overweight,130/85,72,6000,Insomnia
233,Female,44,Teacher,6.6,7,45,4,Overweight,135/90,65,6000,Insomnia
234,Male,44,Salesperson,6.3,6,45,7,Overweight,130/85,72,6000,Insomnia
235,Female,44,Teacher,6.6,7,45,4,Overweight,135/90,65,6000,Insomnia
236,Male,44,Salesperson,6.3,6,45,7,Overweight,130/85,72,6000,Insomnia
237,Male,44,Salesperson,6.4,6,45,7,Overweight,130/85,72,6000,Insomnia
238,Female,44,Teacher,6.5,7,45,4,Overweight,135/90,65,6000,Insomnia
239,Male,44,Salesperson,6.3,6,45,7,Overweight,130/85,72,6000,Insomnia
240,Male,44,Salesperson,6.4,6,45,7,Overweight,130/85,72,6000,Insomnia
241,Female,44,Teacher,6.5,7,45,4,Overweight,135/90,65,6000,Insomnia
242,Male,44,Salesperson,6.3,6,45,7,Overweight,130/85,72,6000,Insomnia
243,Male,44,Salesperson,6.4,6,45,7,Overweight,130/85,72,6000,Insomnia
244,Female,44,Teacher,6.5,7,45,4,Overweight,135/90,65,6000,Insomnia
245,Male,44,Salesperson,6.3,6,45,7,Overweight,130/85,72,6000,Insomnia
246,Female,44,Teacher,6.5,7,45,4,Overweight,135/90,65,6000,Insomnia
247,Male,44,Salesperson,6.3,6,45,7,Overweight,130/85,72,6000,Insomnia
248,Male,44,Engineer,6.8,7,45,7,Overweight,130/85,78,5000,Insomnia
249,Male,44,Salesperson,6.4,6,45,7,Overweight,130/85,72,6000,
250,Male,44,Salesperson,6.5,6,45,7,Overweight,130/85,72,6000,
251,Female,45,Teacher,6.8,7,30,6,Overweight,135/90,65,6000,Insomnia
252,Female,45,Teacher,6.8,7,30,6,Overweight,135/90,65,6000,Insomnia
253,Female,45,Teacher,6.5,7,45,4,Overweight,135/90,65,6000,Insomnia
254,Female,45,Teacher,6.5,7,45,4,Overweight,135/90,65,6000,Insomnia
255,Female,45,Teacher,6.5,7,45,4,Overweight,135/90,65,6000,Insomnia
256,Female,45,Teacher,6.5,7,45,4,Overweight,135/90,65,6000,Insomnia
257,Female,45,Teacher,6.6,7,45,4,Overweight,135/90,65,6000,Insomnia
258,Female,45,Teacher,6.6,7,45,4,Overweight,135/90,65,6000,Insomnia
259,Female,45,Teacher,6.6,7,45,4,Overweight,135/90,65,6000,Insomnia
260,Female,45,Teacher,6.6,7,45,4,Overweight,135/90,65,6000,Insomnia
261,Female,45,Teacher,6.6,7,45,4,Overweight,135/90,65,6000,Insomnia
262,Female,45,Teacher,6.6,7,45,4,Overweight,135/90,65,6000,
263,Female,45,Teacher,6.6,7,45,4,Overweight,135/90,65,6000,
264,Female,45,Manager,6.9,7,55,5,Overweight,125/82,75,5500,
265,Male,48,Doctor,7.3,7,65,5,Obese,142/92,83,3500,Insomnia
266,Female,48,Nurse,5.9,6,90,8,Overweight,140/95,75,10000,Sleep Apnea
267,Male,48,Doctor,7.3,7,65,5,Obese,142/92,83,3500,Insomnia
268,Female,49,Nurse,6.2,6,90,8,Overweight,140/95,75,10000,
269,Female,49,Nurse,6.0,6,90,8,Overweight,140/95,75,10000,Sleep Apnea
270,Female,49,Nurse,6.1,6,90,8,Overweight,140/95,75,10000,Sleep Apnea
271,Female,49,Nurse,6.1,6,90,8,Overweight,140/95,75,10000,Sleep Apnea
272,Female,49,Nurse,6.1,6,90,8,Overweight,140/95,75,10000,Sleep Apnea
273,Female,49,Nurse,6.1,6,90,8,Overweight,140/95,75,10000,Sleep Apnea
274,Female,49,Nurse,6.2,6,90,8,Overweight,140/95,75,10000,Sleep Apnea
275,Female,49,Nurse,6.2,6,90,8,Overweight,140/95,75,10000,Sleep Apnea
276,Female,49,Nurse,6.2,6,90,8,Overweight,140/95,75,10000,Sleep Apnea
277,Male,49,Doctor,8.1,9,85,3,Obese,139/91,86,3700,Sleep Apnea
278,Male,49,Doctor,8.1,9,85,3,Obese,139/91,86,3700,Sleep Apnea
279,Female,50,Nurse,6.1,6,90,8,Overweight,140/95,75,10000,Insomnia
280,Female,50,Engineer,8.3,9,30,3,Normal,125/80,65,5000,
281,Female,50,Nurse,6.0,6,90,8,Overweight,140/95,75,10000,
282,Female,50,Nurse,6.1,6,90,8,Overweight,140/95,75,10000,Sleep Apnea
283,Female,50,Nurse,6.0,6,90,8,Overweight,140/95,75,10000,Sleep Apnea
284,Female,50,Nurse,6.0,6,90,8,Overweight,140/95,75,10000,Sleep Apnea
285,Female,50,Nurse,6.0,6,90,8,Overweight,140/95,75,10000,Sleep Apnea
286,Female,50,Nurse,6.0,6,90,8,Overweight,140/95,75,10000,Sleep Apnea
287,Female,50,Nurse,6.0,6,90,8,Overweight,140/95,75,10000,Sleep Apnea
288,Female,50,Nurse,6.0,6,90,8,Overweight,140/95,75,10000,Sleep Apnea
289,Female,50,Nurse,6.0,6,90,8,Overweight,140/95,75,10000,Sleep Apnea
290,Female,50,Nurse,6.1,6,90,8,Overweight,140/95,75,10000,Sleep Apnea
291,Female,50,Nurse,6.0,6,90,8,Overweight,140/95,75,10000,Sleep Apnea
292,Female,50,Nurse,6.1,6,90,8,Overweight,140/95,75,10000,Sleep Apnea
293,Female,50,Nurse,6.1,6,90,8,Overweight,140/95,75,10000,Sleep Apnea
294,Female,50,Nurse,6.0,6,90,8,Overweight,140/95,75,10000,Sleep Apnea
295,Female,50,Nurse,6.1,6,90,8,Overweight,140/95,75,10000,Sleep Apnea
296,Female,50,Nurse,6.0,6,90,8,Overweight,140/95,75,10000,Sleep Apnea
297,Female,50,Nurse,6.1,6,90,8,Overweight,140/95,75,10000,Sleep Apnea
298,Female,50,Nurse,6.1,6,90,8,Overweight,140/95,75,10000,Sleep Apnea
299,Female,51,Engineer,8.5,9,30,3,Normal,125/80,65,5000,
300,Female,51,Engineer,8.5,9,30,3,Normal,125/80,65,5000,
301,Female,51,Engineer,8.5,9,30,3,Normal,125/80,65,5000,
302,Female,51,Engineer,8.5,9,30,3,Normal,125/80,65,5000,
303,Female,51,Nurse,7.1,7,55,6,Normal Weight,125/82,72,6000,
304,Female,51,Nurse,6.0,6,90,8,Overweight,140/95,75,10000,Sleep Apnea
305,Female,51,Nurse,6.1,6,90,8,Overweight,140/95,75,10000,Sleep Apnea
306,Female,51,Nurse,6.1,6,90,8,Overweight,140/95,75,10000,Sleep Apnea
307,Female,52,Accountant,6.5,7,45,7,Overweight,130/85,72,6000,Insomnia
308,Female,52,Accountant,6.5,7,45,7,Overweight,130/85,72,6000,Insomnia
309,Female,52,Accountant,6.6,7,45,7,Overweight,130/85,72,6000,Insomnia
310,Female,52,Accountant,6.6,7,45,7,Overweight,130/85,72,6000,Insomnia
311,Female,52,Accountant,6.6,7,45,7,Overweight,130/85,72,6000,Insomnia
312,Female,52,Accountant,6.6,7,45,7,Overweight,130/85,72,6000,Insomnia
313,Female,52,Engineer,8.4,9,30,3,Normal,125/80,65,5000,
314,Female,52,Engineer,8.4,9,30,3,Normal,125/80,65,5000,
315,Female,52,Engineer,8.4,9,30,3,Normal,125/80,65,5000,
316,Female,53,Engineer,8.3,9,30,3,Normal,125/80,65,5000,Insomnia
317,Female,53,Engineer,8.5,9,30,3,Normal,125/80,65,5000,
318,Female,53,Engineer,8.5,9,30,3,Normal,125/80,65,5000,
319,Female,53,Engineer,8.4,9,30,3,Normal,125/80,65,5000,
320,Female,53,Engineer,8.4,9,30,3,Normal,125/80,65,5000,
321,Female,53,Engineer,8.5,9,30,3,Normal,125/80,65,5000,
322,Female,53,Engineer,8.4,9,30,3,Normal,125/80,65,5000,
323,Female,53,Engineer,8.4,9,30,3,Normal,125/80,65,5000,
324,Female,53,Engineer,8.5,9,30,3,Normal,125/80,65,5000,
325,Female,53,Engineer,8.3,9,30,3,Normal,125/80,65,5000,
326,Female,53,Engineer,8.5,9,30,3,Normal,125/80,65,5000,
327,Female,53,Engineer,8.3,9,30,3,Normal,125/80,65,5000,
328,Female,53,Engineer,8.5,9,30,3,Normal,125/80,65,5000,
329,Female,53,Engineer,8.3,9,30,3,Normal,125/80,65,5000,
330,Female,53,Engineer,8.5,9,30,3,Normal,125/80,65,5000,
331,Female,53,Engineer,8.5,9,30,3,Normal,125/80,65,5000,
332,Female,53,Engineer,8.4,9,30,3,Normal,125/80,65,5000,
333,Female,54,Engineer,8.4,9,30,3,Normal,125/80,65,5000,
334,Female,54,Engineer,8.4,9,30,3,Normal,125/80,65,5000,
335,Female,54,Engineer,8.4,9,30,3,Normal,125/80,65,5000,
336,Female,54,Engineer,8.4,9,30,3,Normal,125/80,65,5000,
337,Female,54,Engineer,8.4,9,30,3,Normal,125/80,65,5000,
338,Female,54,Engineer,8.4,9,30,3,Normal,125/80,65,5000,
339,Female,54,Engineer,8.5,9,30,3,Normal,125/80,65,5000,
340,Female,55,Nurse,8.1,9,75,4,Overweight,140/95,72,5000,Sleep Apnea
341,Female,55,Nurse,8.1,9,75,4,Overweight,140/95,72,5000,Sleep Apnea
342,Female,56,Doctor,8.2,9,90,3,Normal Weight,118/75,65,10000,
343,Female,56,Doctor,8.2,9,90,3,Normal Weight,118/75,65,10000,
344,Female,57,Nurse,8.1,9,75,3,Overweight,140/95,68,7000,
345,Female,57,Nurse,8.2,9,75,3,Overweight,140/95,68,7000,Sleep Apnea
346,Female,57,Nurse,8.2,9,75,3,Overweight,140/95,68,7000,Sleep Apnea
347,Female,57,Nurse,8.2,9,75,3,Overweight,140/95,68,7000,Sleep Apnea
348,Female,57,Nurse,8.2,9,75,3,Overweight,140/95,68,7000,Sleep Apnea
349,Female,57,Nurse,8.2,9,75,3,Overweight,140/95,68,7000,Sleep Apnea
350,Female,57,Nurse,8.1,9,75,3,Overweight,140/95,68,7000,Sleep Apnea
351,Female,57,Nurse,8.1,9,75,3,Overweight,140/95,68,7000,Sleep Apnea
352,Female,57,Nurse,8.1,9,75,3,Overweight,140/95,68,7000,Sleep Apnea
353,Female,58,Nurse,8.0,9,75,3,Overweight,140/95,68,7000,Sleep Apnea
354,Female,58,Nurse,8.0,9,75,3,Overweight,140/95,68,7000,Sleep Apnea
355,Female,58,Nurse,8.0,9,75,3,Overweight,140/95,68,7000,Sleep Apnea
356,Female,58,Nurse,8.0,9,75,3,Overweight,140/95,68,7000,Sleep Apnea
357,Female,58,Nurse,8.0,9,75,3,Overweight,140/95,68,7000,Sleep Apnea
358,Female,58,Nurse,8.0,9,75,3,Overweight,140/95,68,7000,Sleep Apnea
359,Female,59,Nurse,8.0,9,75,3,Overweight,140/95,68,7000,
360,Female,59,Nurse,8.1,9,75,3,Overweight,140/95,68,7000,
361,Female,59,Nurse,8.2,9,75,3,Overweight,140/95,68,7000,Sleep Apnea
362,Female,59,Nurse,8.2,9,75,3,Overweight,140/95,68,7000,Sleep Apnea
363,Female,59,Nurse,8.2,9,75,3,Overweight,140/95,68,7000,Sleep Apnea
364,Female,59,Nurse,8.2,9,75,3,Overweight,140/95,68,7000,Sleep Apnea
365,Female,59,Nurse,8.0,9,75,3,Overweight,140/95,68,7000,Sleep Apnea
366,Female,59,Nurse,8.0,9,75,3,Overweight,140/95,68,7000,Sleep Apnea
367,Female,59,Nurse,8.1,9,75,3,Overweight,140/95,68,7000,Sleep Apnea
368,Female,59,Nurse,8.0,9,75,3,Overweight,140/95,68,7000,Sleep Apnea
369,Female,59,Nurse,8.1,9,75,3,Overweight,140/95,68,7000,Sleep Apnea
370,Female,59,Nurse,8.1,9,75,3,Overweight,140/95,68,7000,Sleep Apnea
371,Female,59,Nurse,8.0,9,75,3,Overweight,140/95,68,7000,Sleep Apnea
372,Female,59,Nurse,8.1,9,75,3,Overweight,140/95,68,7000,Sleep Apnea
373,Female,59,Nurse,8.1,9,75,3,Overweight,140/95,68,7000,Sleep Apnea
374,Female,59,Nurse,8.1,9,75,3,Overweight,140/95,68,7000,Sleep Apnea
'''

# Create DataFrame from string data
df_cleaned = pd.read_csv(StringIO(data))

# Drop Person ID as it's not needed
df_cleaned = df_cleaned.drop('Person ID', axis=1)

# Split Blood Pressure into Systolic and Diastolic
df_cleaned[['Systolic_BP', 'Diastolic_BP']] = df_cleaned['Blood Pressure'].str.split('/', expand=True).astype(int)
df_cleaned = df_cleaned.drop('Blood Pressure', axis=1)

# Handle missing values in Sleep Disorder
df_cleaned['Sleep Disorder'] = df_cleaned['Sleep Disorder'].fillna('None')

# Create dummy variables
df_encoded = pd.get_dummies(df_cleaned, drop_first=True)
X = df_encoded.drop(columns=['Stress Level'])
y = df_encoded['Stress Level']

# Store feature columns in session state
st.session_state.feature_columns = X.columns

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model and store in session state
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
st.session_state.model = rf_model

# Get unique occupations
unique_occupations = df_cleaned['Occupation'].unique().tolist()
unique_occupations.append("Others")

# Rest of your code remains exactly the same...

# Rest of your code remains the same...
# (Custom CSS styling, main functions, etc.)

# Custom CSS styling
st.markdown("""
    <style>
    .main {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stButton>button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .chat-container {
        max-width: 800px;
        margin: auto;
        padding: 20px;
        border: 1px solid #4ECDC4;
        border-radius: 15px;
        margin-bottom: 20px;
        background: rgba(255,255,255,0.05);
    }
    .chat-message {
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        display: flex;
        align-items: flex-start;
    }
    .user-message {
        background: linear-gradient(45deg, #FF6B6B22, #FF6B6B44);
        margin-left: 50px;
    }
    .bot-message {
        background: linear-gradient(45deg, #4ECDC422, #4ECDC444);
        margin-right: 50px;
    }
    .advice-box {
        padding: 20px;
        border-radius: 15px;
        background: linear-gradient(45deg, #2C3E5022, #3498DB22);
        margin: 20px 0;
        border: 1px solid #3498DB;
    }
    .footer {
        background: linear-gradient(45deg, #2196F3, #64B5F6);
        padding: 30px;
        border-radius: 20px;
        margin-top: 50px;
        color: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(33, 150, 243, 0.5);
    }
    </style>
""", unsafe_allow_html=True)

st.title("Welcome to Mental Stress Manager")
st.markdown("""
This app is your personal AI companion for managing stress and mental wellbeing.

### Features:
- Structured questionnaire for stress assessment
- AI-powered predictions
- Detailed personalized recommendations
- Holistic wellness advice
""")

def get_detailed_advice(stress_level, user_data):
    age = int(user_data['age'])
    sleep_quality = int(user_data['sleep_quality'])
    activity_level = int(user_data['activity_level'])
    
    advice = {
        "High": {
            "Immediate Actions": [
                "Take deep breaths for 5 minutes every hour",
                "Step away from stressful situations when possible",
                "Practice the 5-4-3-2-1 grounding technique"
            ],
            "Daily Practices": [
                f"Given your activity level of {activity_level}/10, gradually increase physical activity",
                f"With your sleep quality at {sleep_quality}/10, focus on sleep hygiene",
                "Maintain a stress journal to identify triggers",
                "Practice progressive muscle relaxation before bed"
            ],
            "Long-term Strategies": [
                "Consider professional counseling or therapy",
                "Join stress management workshops",
                "Build a support network",
                "Learn time management techniques"
            ],
            "Lifestyle Modifications": [
                "Reduce caffeine and processed foods",
                "Create a calming morning routine",
                "Set boundaries in work and personal life",
                "Take up a relaxing hobby like gardening or painting"
            ]
        },
        "Medium": {
            "Daily Practices": [
                "15-minute morning meditation",
                f"Based on your age ({age}), appropriate exercise routine",
                "Regular breaks during work",
                "Nature walks or outdoor time"
            ],
            "Wellness Tips": [
                "Practice mindful eating",
                "Maintain a gratitude journal",
                "Regular stretching exercises",
                "Digital detox for 1 hour before bed"
            ],
            "Preventive Measures": [
                "Set realistic goals and priorities",
                "Create a balanced weekly schedule",
                "Practice saying 'no' when necessary",
                "Regular social connections"
            ]
        },
        "Low": {
            "Maintenance Tips": [
                "Continue your effective stress management practices",
                "Regular exercise and movement",
                "Maintain social connections",
                "Healthy sleep schedule"
            ],
            "Enhancement Strategies": [
                "Set new personal growth goals",
                "Learn new skills or hobbies",
                "Share your successful strategies with others",
                "Regular wellness check-ins"
            ]
        }
    }
    return advice[stress_level]

def predict_stress(user_data):
    if st.session_state.model is None or st.session_state.feature_columns is None:
        st.error("Model not initialized properly")
        return None, None

    input_data = pd.DataFrame({
        'Age': [user_data['age']],
        'Sleep Duration': [user_data['sleep_duration']],
        'Quality of Sleep': [user_data['sleep_quality']],
        'Physical Activity Level': [user_data['activity_level']],
        'Heart Rate': [user_data['heart_rate']],
        'Daily Steps': [user_data['daily_steps']],
        'Gender_Male': [1 if user_data['gender'].lower().strip() == "male" else 0],
        'Systolic_BP': [user_data['systolic_bp']],
        'Diastolic_BP': [user_data['diastolic_bp']]
    })

    # Handle occupation encoding
    if user_data['occupation'].strip().title() == "Others":
        for occ in unique_occupations:
            input_data[f"Occupation_{occ}"] = [0]
    else:
        for occ in unique_occupations:
            input_data[f"Occupation_{occ}"] = [1 if user_data['occupation'].strip().title() == occ else 0]

    # Encode BMI category
    input_data[f'BMI Category_Normal'] = [1 if user_data['bmi_category'].strip().title() == "Normal" else 0]
    input_data[f'BMI Category_Overweight'] = [1 if user_data['bmi_category'].strip().title() == "Overweight" else 0]
    input_data[f'BMI Category_Obese'] = [1 if user_data['bmi_category'].strip().title() == "Obese" else 0]

    # Encode Sleep Disorder
    input_data[f'Sleep Disorder_None'] = [1 if user_data['sleep_disorder'].strip().title() == "None" else 0]
    input_data[f'Sleep Disorder_Sleep Apnea'] = [1 if user_data['sleep_disorder'].strip().title() == "Sleep Apnea" else 0]
    input_data[f'Sleep Disorder_Insomnia'] = [1 if user_data['sleep_disorder'].strip().title() == "Insomnia" else 0]

    # Reorder columns to match training data
    input_data = input_data.reindex(columns=st.session_state.feature_columns, fill_value=0)
    
    prediction = st.session_state.model.predict(input_data)[0]
    
    if prediction > 7:
        return "High", prediction
    elif prediction > 4:
        return "Medium", prediction
    else:
        return "Low", prediction

def main():
    st.markdown("<h1>Mental Stress Assessment</h1>", unsafe_allow_html=True)
    
    questions = [
        ("Please enter your gender (Male/Female):", "gender", lambda x: x.strip().lower() in ["male", "female"]),
        ("What is your age?", "age", lambda x: x.strip().isdigit() and 18 <= int(x) <= 100),
        (f"What is your occupation? ({', '.join(unique_occupations)}):", "occupation", lambda x: x.strip().title() in unique_occupations),
        ("How many hours do you sleep per day? (4-12)", "sleep_duration", lambda x: x.strip().replace('.','',1).isdigit() and 4 <= float(x) <= 12),
        ("Rate your sleep quality (1-10):", "sleep_quality", lambda x: x.strip().isdigit() and 1 <= int(x) <= 10),
        ("Rate your physical activity level (1-10):", "activity_level", lambda x: x.strip().isdigit() and 1 <= int(x) <= 10),
        ("Enter your BMI category (Normal/Overweight/Obese):", "bmi_category", lambda x: x.strip().title() in ["Normal", "Overweight", "Obese"]),
        ("Enter your systolic blood pressure (90-200):", "systolic_bp", lambda x: x.strip().isdigit() and 90 <= int(x) <= 200),
        ("Enter your diastolic blood pressure (60-130):", "diastolic_bp", lambda x: x.strip().isdigit() and 60 <= int(x) <= 130),
        ("Enter your heart rate (60-120):", "heart_rate", lambda x: x.strip().isdigit() and 60 <= int(x) <= 120),
        ("Enter your daily steps (1000-20000):", "daily_steps", lambda x: x.strip().isdigit() and 1000 <= int(x) <= 20000),
        ("Do you have any sleep disorder? (None/Sleep Apnea/Insomnia):", "sleep_disorder", lambda x: x.strip().title() in ["None", "Sleep Apnea", "Insomnia"])
    ]

    # Display chat history
    for i, (q, a) in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"""
                <div class='chat-container'>
                    <div class='chat-message bot-message'>{q}</div>
                    <div class='chat-message user-message'>{a}</div>
                </div>
            """, unsafe_allow_html=True)

    if st.session_state.step < len(questions):
        question, key, validator = questions[st.session_state.step]
        user_input = st.text_input(question, key=f"input_{st.session_state.step}")
        
        col1, col2 = st.columns(2)
        with col1:
            next_button = st.button("Next", key="next_button")
            if next_button and not st.session_state.get('next_clicked', False):
                st.session_state.next_clicked = True
                if validator(user_input):
                    st.session_state.user_data[key] = user_input.strip()
                    st.session_state.chat_history.append((question, user_input.strip()))
                    st.session_state.step += 1
                else:
                    st.error("Please provide a valid input")
                st.session_state.next_clicked = False
        with col2:
            clear_button = st.button("Clear", key="clear_button")
            if clear_button and not st.session_state.get('clear_clicked', False):
                st.session_state.clear_clicked = True
                st.session_state.step = 0
                st.session_state.user_data = {}
                st.session_state.chat_history = []
                st.session_state.clear_clicked = False
                
    else:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Get Detailed Assessment"):
                level, score = predict_stress(st.session_state.user_data)
                if level is not None and score is not None:
                    detailed_advice = get_detailed_advice(level, st.session_state.user_data)
                    
                    st.markdown(f"""
                        <div class='advice-box'>
                            <h2>Your Stress Assessment</h2>
                            <h3>Stress Level: {level} ({score:.1f}/10)</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    for category, tips in detailed_advice.items():
                        st.markdown(f"""
                            <div class='advice-box'>
                                <h3>{category}</h3>
                                <ul>
                                    {"".join([f"<li>{tip}</li>" for tip in tips])}
                                </ul>
                            </div>
                        """, unsafe_allow_html=True)
        with col2:
            if st.button("Start Over"):
                st.session_state.step = 0
                st.session_state.user_data = {}
                st.session_state.chat_history = []
                
    # Professional Footer with Mobile-Friendly Design
    st.markdown("""
        <style>
            .professional-footer {
                background: linear-gradient(135deg, #2b5876 0%, #4e4376 100%);
                color: white;
                padding: 1.5rem;
                border-radius: 10px;
                margin-top: 1.5rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                width: 100%;
            }
            .team-info h3 {
                color: #ffffff;
                font-size: 1.5rem;
                margin-bottom: 1rem;
                text-align: center;
                font-weight: bold;
            }
            .team-grid {
                display: flex;
                flex-direction: column;
                gap: 1rem;
                padding: 0.5rem;
            }
            .team-lead, .team-members {
                background: rgba(255, 255, 255, 0.1);
                padding: 1rem;
                border-radius: 8px;
                backdrop-filter: blur(5px);
                width: 100%;
            }
            .team-lead h4, .team-members h4 {
                color: #ffd700;
                margin-bottom: 0.5rem;
                font-size: 1.2rem;
                font-weight: bold;
            }
            .team-members ul {
                list-style: none;
                padding: 0;
                margin: 0;
            }
            .team-members li {
                margin: 0.3rem 0;
                color: #ffffff;
                font-size: 1rem;
                font-weight: bold;
            }
            .team-lead p {
                font-size: 1rem;
                font-weight: bold;
                margin: 0.2rem 0;
            }
            .team-description {
                font-size: 0.9rem;
                color: #e0e0e0;
                margin: 0.2rem 0;
                font-style: italic;
            }
            .copyright {
                text-align: center;
                margin-top: 1.5rem;
                padding-top: 1rem;
                border-top: 1px solid rgba(255, 255, 255, 0.2);
                font-size: 0.9rem;
            }
            .made-with-love {
                color: #ffd700;
                font-weight: bold;
                margin-top: 0.5rem;
                font-size: 1rem;
            }
            .contact-info {
                text-align: center;
                margin-top: 1rem;
                color: #ffffff;
                font-weight: bold;
                font-size: 0.9rem;
            }
            /* Responsive Design */
            @media screen and (min-width: 768px) {
                .team-grid {
                    flex-direction: row;
                    justify-content: space-around;
                }
                .team-lead, .team-members {
                    width: 45%;
                }
                .team-info h3 {
                    font-size: 2rem;
                }
                .team-lead h4, .team-members h4 {
                    font-size: 1.5rem;
                }
                .team-members li, .team-lead p {
                    font-size: 1.2rem;
                }
                .team-description {
                    font-size: 1rem;
                }
                .contact-info, .copyright {
                    font-size: 1.1rem;
                }
            }
            /* Touch-friendly improvements */
            .team-lead, .team-members {
                touch-action: manipulation;
                -webkit-tap-highlight-color: transparent;
            }
            /* Better readability in different modes */
            @media (prefers-color-scheme: dark) {
                .professional-footer {
                    background: linear-gradient(135deg, #1a1a1a 0%, #4e4376 100%);
                }
            }
            @media (prefers-color-scheme: light) {
                .professional-footer {
                    background: linear-gradient(135deg, #2b5876 0%, #4e4376 100%);
                }
            }
        </style>
        <footer class='professional-footer'>
            <div class='team-info'>
                <h3>Meet our Exceptional Development Team</h3>
                <div class='team-grid'>
                    <div class='team-lead'>
                        <h4>Project Lead</h4>
                        <p>Vikhram S</p>
                        <p class='team-description'>Lead ML Engineer</p>
                        <p class='team-description'>• Developed core ML algorithms</p>
                        <p class='team-description'>• Implemented Streamlit frontend</p>
                        <p class='team-description'>• Designed system architecture</p>
                    </div>
                    <div class='team-members'>
                        <h4>Co-Developers</h4>
                        <ul>
                            <li>Ragul S</li>
                            <p class='team-description'>• Data preprocessing & Feature engineering</p>
                            <li>Roshan R</li>
                            <p class='team-description'>• Model testing & Validation</p>
                            <li>Nithesh Kumar B</li>
                            <p class='team-description'>• Documentation & Testing</p>
                        </ul>
                    </div>
                </div>
            </div>
            <div class='contact-info'>
                <p>For Customer Support & Technical Inquiries:</p>
                <p>vikhrams@saveetha.ac.in</p>
            </div>
            <div class='copyright'>
                <p>© 2024 Mental Stress Manager by Z Data Knights. All Rights Reserved.</p>
                <p class='made-with-love'>Made With ❤️ by Team Z Data Knights</p>
            </div>
        </footer>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
