import os
import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler as SS
from sklearn.linear_model import LogisticRegression as LogR


TASKS = ['aic', 'logr', 'logr_eusipco']

def get_inputs_outputs(features_path, gt_path):    

	features = np.load(features_path)
	max_linear = features['max_linear']
	max_db = features['max_db']
	peak_loc = features['peak_loc']
	gt = np.load(gt_path)
	
	return max_linear.reshape(-1,1), max_db.reshape(-1,1), peak_loc.reshape(-1, 1), gt

def AIC(k, y_label, ypred):

	# number of samples
	n = len(y_label)
	
	# sigma^2
	sigma_sq = np.sum((y_label-ypred)**2)
	
	# AICc
	return n*np.log(sigma_sq) + 2*k

def compute_aic_min(max_linear, max_db, peak_loc, gt):
	# Stack max_db and peak_loc to one input
	maxdb_peakloc = np.hstack([max_db, peak_loc])

	# Scaling
	x_max_linear = SS().fit_transform(max_linear)
	x_max_db = SS().fit_transform(max_db)
	x_peak_loc = SS().fit_transform(peak_loc)
	x_maxdb_peakloc = SS().fit_transform(maxdb_peakloc)

	classifiers = [
		('logistic regression', LogR()),
	]

	inputs = [
		('max_linear', x_max_linear),
		('max_db', x_max_db),
		('peak_loc', x_peak_loc),
		('max_db + peak_loc', x_maxdb_peakloc)
	]

	AICs = []
	models = []
	modelnames = []
	f1_scores = []
	w1 = []

	for clname, clf in classifiers:
		for inpname, x in inputs:
			name = "{}, {}".format(clname, inpname)
			modelnames.append(name)
			
			# Fit classifier
			clf.fit(x, gt)
			models.append(clf)
			
			# Predict new data
			ypred = clf.predict(x)
			
			# Free parameters are the coefficients + intercept
			k = len(clf.coef_) + 1

			# Compute AIC
			aic = AIC(k, gt, ypred)
			AICs.append(aic)
			
			# Show f1
			f1 = f1_score(gt, ypred)
			f1_scores.append(f1)
			
	n_best = int(np.argmin(AICs))
	DAICs = [a - min(AICs) for a in AICs]
	daics_array = np.array(DAICs)
	w_denom = np.sum(np.exp(-0.5*daics_array))

	for n in range(len(DAICs)):
		w1.append(np.exp(-0.5*DAICs[n])/w_denom)

	records = []
	for n in range(len(AICs)):
		record = {
			'name': modelnames[n],
			'AIC' : "{:0.2f}".format(AICs[n]),
			'DAIC': "{:0.2f}".format(daics_array[n]),
			'w'   : "{:0.2f}".format(w1[n]),
			'f1'  : "{:0.2f}".format(f1_scores[n])
		}
		records.append(record)

	results_df = pd.DataFrame.from_records(records, columns=['name', 'AIC', 'DAIC', 'w', 'f1'])
	
	print("Model with least AIC: {}".format(modelnames[n_best]))
	print(results_df)

def compute_logr(max_linear, max_db, peak_loc, gt):
	# Stack max_db and peak_loc to one input
	maxdb_peakloc = np.hstack([max_db, peak_loc])
	
	# Scaling
	x_maxdb_peakloc = SS().fit_transform(maxdb_peakloc)

	classifiers = [
		('logistic regression', LogR()),
	]

	inputs = [
		('max_db + peak_loc', x_maxdb_peakloc),
	]

	models = []
	modelnames = []
	p_scores = []
	r_scores = []
	f1_scores = []
	f1micro_scores = []

	for clname, clf in classifiers:
		for inpname, x in inputs:
			name = "{}, {}".format(clname, inpname)
			modelnames.append(name)

			# shuffle and split training and test sets
			X_train, X_test, y_train, y_test = train_test_split(x, gt, test_size=.5,random_state=0)

			# Fit classifier
			clf.fit(X_train, y_train)
			models.append(clf)

			# Predict new data
			ypred = clf.predict(X_test)

			# Show the precision, recall and fscore for label 1
			precision,recall,fscore,support = precision_recall_fscore_support(y_test, ypred)
			p_scores.append(precision[1])
			r_scores.append(recall[1])
			f1_scores.append(fscore[1])

			# Show micro-f1 as overall accuracy
			f1 = f1_score(y_test, ypred, average='micro')
			f1micro_scores.append(f1)


	records = []
	for n in range(len(f1micro_scores)):
		record = {
			'name': modelnames[n],
			'precision' : "{:0.2f}".format(p_scores[n]),
			'recall': "{:0.2f}".format(r_scores[n]),
			'f1' : "{:0.2f}".format(f1_scores[n]),
			'f1micro' : "{:0.2f}".format(f1micro_scores[n])
		}
		records.append(record)

	results_df = pd.DataFrame.from_records(records, columns=['precision', 'recall', 'f1', 'f1micro'])
	print(results_df)

def compute_logr_eusipco(eusipco_dir):
	for piece in ['chopin','brahms','ravel','beethoven']:
		features_path = os.path.join(eusipco_dir, "{}_features.npz".format(piece))
		gt_path = os.path.join(eusipco_dir, "{}_gt.npy".format(piece))
		max_linear, max_db, peak_loc, gt = get_inputs_outputs(features_path, gt_path)

		# Stack max_db and peak_loc to one input
		maxdb_peakloc = np.hstack([max_db, peak_loc])
		
		# Scaling
		x_maxdb_peakloc = SS().fit_transform(maxdb_peakloc)

		classifiers = [
			('balanced logistic regression', LogR(random_state=0,class_weight='balanced')),
			('logistic regression', LogR()),
		]

		inputs = [
			('max_db + peak_loc', x_maxdb_peakloc),
		]

		models = []
		modelnames = []
		p_scores = []
		r_scores = []
		f1_scores = []
		f1micro_scores = []
		clnames = []

		for clname, clf in classifiers:
			for inpname, x in inputs:
				name = "{}, {}".format(clname, inpname)
				modelnames.append(name)

				# shuffle and split training and test sets
				X_train, X_test, y_train, y_test = train_test_split(x, gt, test_size=.5,random_state=0)

				# Fit classifier
				clf.fit(X_train, y_train)
				models.append(clf)

				# Predict new data
				ypred = clf.predict(X_test)

				# Show the precision, recall and fscore for label 1
				precision,recall,fscore,support = precision_recall_fscore_support(y_test, ypred)
				p_scores.append(precision[1])
				r_scores.append(recall[1])
				f1_scores.append(fscore[1])

				# Show micro-f1 as overall accuracy
				f1 = f1_score(y_test, ypred, average='micro')
				f1micro_scores.append(f1)

				clnames.append(clname)


		records = []
		for n in range(len(f1micro_scores)):
			record = {
				'clname':clnames[n],
				'piece': piece,
				'name': modelnames[n],
				'precision' : "{:0.2f}".format(p_scores[n]),
				'recall': "{:0.2f}".format(r_scores[n]),
				'f1' : "{:0.2f}".format(f1_scores[n]),
				'f1micro' : "{:0.2f}".format(f1micro_scores[n])
			}
			records.append(record)

		results_df = pd.DataFrame.from_records(records, columns=['clname', 'piece', 'precision', 'recall', 'f1', 'f1micro'])
		print(results_df)


def main(args):
	if args.task not in TASKS:
		raise ValueError("task must be one of {}".format(TASKS))

	if args.task == 'logr_eusipco':
		compute_logr_eusipco('eusipco-data')

	else:
		max_linear, max_db, peak_loc, gt = get_inputs_outputs(args.features_path, args.gt_path)

		if args.task == 'aic':
			compute_aic_min(max_linear, max_db, peak_loc, gt)
		elif args.task == 'logr':
			compute_logr(max_linear, max_db, peak_loc, gt)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Evaluate for legato-pedal onset detection")
	parser.add_argument("task",
						type=str,
						help="Task to compute one of "
						"aic, logr, logr_eusipco.")
	parser.add_argument("features_path",
						type=str,
						default = "features/chopin_features.npz",
						help="Path to file for features"
						"including max_linear, max_db and peak_loc.")
	parser.add_argument("gt_path",
						type=str,
						default = "input/chopin/chopin_gt.npy",
						help="Path to file for ground truth")


	main(parser.parse_args())