import pandas as pd
import numpy as np
import pickle
from sklearn.mixture import GaussianMixture

def get_feats(data):
	feats=[]
	for i in range(0,len(data)):
		instance=data['feature'][i].tolist()
		feats.append(instance)
	feat1=pd.DataFrame([subject for subject in feats])
	return feat1

unique_labels=[]
lines=open("phone.txt",'r')
for l in lines:
	unique_labels.append((l.split('\n'))[0])
	os.makedirs('models/'+, exist_ok=True)

for phoneme in unique_labels:
	fname=str(phoneme)+"_train.hdf"
	dataframe=pd.read_hdf("./features_seperated/mfcc/"+fname)
	dataframe_delta=pd.read_hdf("./features_seperated/mfcc_delta/"+fname)
	dataframe_delta_delta=pd.read_hdf("./features_seperated/mfcc_delta_delta/"+fname)
	feats=get_feats(dataframe)
	feats_d=get_feats(dataframe_delta)
	feats_dd=get_feats(dataframe_delta_delta)
	for mixtur in range(1,9):
		mixture=pow(2,mixtur)
		model_name="energy_mfcc"+str(mixture)+".pkl"
		model_name_w="mfcc"+str(mixture)+".pkl"
		model=GaussianMixture(n_components=mixture, covariance_type='diag')
		model.fit(feats.as_matrix(columns=None))
		model_w=GaussianMixture(n_components=mixture, covariance_type='diag')
		model_w.fit((feats.drop(columns=[0],axis=1)).as_matrix(columns=None))
		pickle.dump(model,open("./models/"+str(phoneme)+"/"+model_name, 'wb'))
		pickle.dump(model_w, open("./models/"+str(phoneme)+"/"+model_name_w, 'wb'))


	model_delta_name="energy_delta.pkl"
	model_delta_name_w="delta.pkl"
	model_dd_name="energy_delta_delta.pkl"
	model_dd_name_w="delta_delta.pkl"
	model1=GaussianMixture(n_components=64, covariance_type='diag')
	model1.fit(feats_d.as_matrix(columns=None))
	pickle.dump(model1, open("./models/"+str(phoneme)+"/"+model_delta_name, 'wb'))
	model1_w=GaussianMixture(n_components=64,covariance_type='diag')
	model1_w.fit((feats_d.drop(columns=[0,13],axis=1)).as_matrix(columns=None))
	pickle.dump(model1_w, open("./models/"+str(phoneme)+"/"+model_delta_name_w, 'wb'))
	model2=GaussianMixture(n_components=64, covariance_type='diag')
	model2.fit(feats_dd.as_matrix(columns=None))
	pickle.dump(model2, open("./models/"+str(phoneme)+"/"+model_dd_name, 'wb'))
	model2_w=GaussianMixture(n_components=64, covariance_type='diag')
	model2_w.fit((feature_dd.drop(columns=[0,13,26],axis=1)).as_matrix(columns=None))
	pickle.dump(model2_w, open("./models/"+str(phoneme)+"/"+model_dd_name_w, 'wb'))




#Training code
