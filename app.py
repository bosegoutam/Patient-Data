from flask import Flask, render_template, request
from wtforms import Form, DecimalField, SelectField, validators
import numpy as np
import os
import pandas as pd
import pickle

app = Flask(__name__)

######## Preparing the Classifier
cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir,'pkl_objects','classifier.pkl'), 'rb'))

def classify(df):
	label = {0:'Admission not required', 1:'Admission required'}
	y = clf.predict(df)[0]
	proba = np.max(clf.predict_proba(df))
	return label[y], proba

class PatientForm(Form):
	age = DecimalField('Age',[validators.DataRequired(message='Please enter valid age'), validators.NumberRange(min=0, max=175)])
	gender = SelectField(u'Gender', choices=[('F', 'Female'), ('M', 'Male'), ('O', 'Other')])
	hypertension = SelectField(u'Hypertension', choices=[('0', 'No'), ('1', 'Yes')])
	heart = SelectField(u'Heart Disease', choices=[('0', 'No'), ('1', 'Yes')])
	married = SelectField(u'Ever Married', choices=[('1', 'Yes'), ('0', 'No')])
	work = SelectField(u'Work Type', choices=[('G', 'Govt. Job'), ('N', 'Never Worked'),  ('P', 'Private'),  ('C', 'Child'), ('S', 'Self-Employed')])
	residence = SelectField(u'Residence Type', choices=[('R', 'Rural'), ('U', 'Urban')])
	ppbs = DecimalField('Average Glucose Level',[validators.DataRequired(message='Please enter valid PPBS value'), validators.NumberRange(min=30, max=750)])
	bmi = DecimalField('BMI',[validators.DataRequired(message='Please enter valid BMI value'), validators.NumberRange(min=7, max=100)])
	smoking = SelectField(u'Smoking Status', choices=[('N', 'Never Smoked'), ('F', 'Formerly Smoked'), ('S', 'Smokes')])

@app.route('/')
def index():
	form = PatientForm(request.form)
	return render_template('patient.html', form=form)

@app.route('/result', methods=['POST'])
def result():
	form = PatientForm(request.form)
	if request.method == 'POST' and form.validate():
		age = request.form['age']
		gender = request.form['gender']
		hypertension = request.form['hypertension']
		heart = request.form['heart']
		married = request.form['married']
		work = request.form['work']
		residence = request.form['residence']
		ppbs = request.form['ppbs']
		bmi = request.form['bmi']
		smoking = request.form['smoking']
		
		if gender == 'F':
			female = 1
			male = 0
			other = 0
		elif gender == 'M':
			female = 0
			male = 1
			other = 0
		elif gender == 'O':
			female = 0
			male = 0
			other = 1
			
		if married == '1':	
			m_ys = 1
			m_no = 0
		elif married == '0':	
			m_ys = 0
			m_no = 1
			
		if work == 'C':
			w_c = 1
			w_g = 0
			w_n = 0
			w_p = 0
			w_s = 0
		elif work == 'G':
			w_c = 0
			w_g = 1
			w_n = 0
			w_p = 0
			w_s = 0
		elif work == 'N':
			w_c = 0
			w_g = 0
			w_n = 1
			w_p = 0
			w_s = 0
		elif work == 'P':
			w_c = 0
			w_g = 0
			w_n = 0
			w_p = 1
			w_s = 0
		elif work == 'S':
			w_c = 0
			w_g = 0
			w_n = 0
			w_p = 0
			w_s = 1
			
		if residence == 'R':	
			rural = 1
			urban = 0
		elif residence == 'U':	
			rural = 0
			urban = 1
		
		if smoking == 'N':
			s_n = 1
			s_f = 0
			s_s = 0
		elif smoking == 'F':
			s_n = 0
			s_f = 1
			s_s = 0
		elif smoking == 'S':
			s_n = 0
			s_f = 0
			s_s = 1
			
		d = {'age': [float(age)], 'hypertension': [int(hypertension)], 'heart_disease': [int(heart)], 'avg_glucose_level': [float(ppbs)], \
		'bmi': [float(bmi)], 'gender_Female': [female], 'gender_Male': [male], 'gender_Other': [other], \
		'ever_married_No': [m_no], 'ever_married_Yes': [m_ys], 'work_type_Govt_job': [w_g], \
		'work_type_Never_worked': [w_n], 'work_type_Private': [w_p], 'work_type_Self-employed': [w_s], 'work_type_children': [w_c], \
		'residence_type_Rural': [rural], 'residence_type_Urban': [urban], 'smoking_status_formerly smoked': [s_f], \
		'smoking_status_never smoked': [s_n], 'smoking_status_smokes': [s_s]}
		
		pd.set_option('display.max_rows', 500)
		pd.set_option('display.max_columns', 500)
		pd.set_option('display.width', 1000)
		
		X = pd.DataFrame(data=d)
		print(X.head())
		y, proba = classify(X)
		return render_template('result.html', prediction=y, probability=round(proba*100, 2))
	return render_template('patient.html', form=form)

if __name__ == '__main__':
  app.run(debug=True)