"""Script to detect legato-pedal onset from piano audio"""

import os
import argparse
import numpy as np
import scipy
from scipy.stats import mode
import librosa
np.seterr(divide='ignore', invalid='ignore')

SR = 44100
N = 2048
HOP_SIZE = 512
FREQ_RESO = (SR/2.0)/(N/2+1)
FRM_RESO = (1.0/SR)*HOP_SIZE
EPS = np.finfo(float).eps


def get_rmse(H, note_events, f0, B1, seg_starts, tempo, offTime):

	S, phase = librosa.magphase(H)
	minS = np.min(S)
	S_nopartial = np.copy(S)
	note_onsets = note_events[:,0]
	note_offsets = note_events[:,1]
	note_inds = note_events[:,2]-21
	
	# Remove the partials from H based on transcription results
	for i,ind in enumerate(note_inds):
		f0freq = f0[int(ind)]
		inhar = B1[int(ind)]
		nH = int(np.floor(((-1+(1+4*inhar*((SR/3/f0freq)**2))**0.5)/(2*inhar))**0.5))
		onsetT = note_onsets[i]
		offsetT = note_offsets[i]
		hfs = np.arange(1, nH+1)*f0freq*(1+inhar*np.arange(1, nH+1)**2)**0.5

		hfs_freqInds = hfs/FREQ_RESO
		hfs_frmStartIndx = int(np.floor(onsetT/FRM_RESO))
		hfs_frmEndIndx = int(np.floor(offsetT/FRM_RESO))+1

		for hfs_freqInd in hfs_freqInds:
			hfs_freqInd = int(np.floor(hfs_freqInd))
			S_nopartial[hfs_freqInd,hfs_frmStartIndx:hfs_frmEndIndx] = minS

	# Get rms features for each segment
	rmse_feature = np.zeros(S_nopartial.shape[1]) 
	for i,t in enumerate(seg_starts):
		segStartT = t
		if t == seg_starts[-1]:
			segEndT = offTime
		else:
			segEndT = seg_starts[i+1]

		frmStartIndx = int(np.floor(segStartT/FRM_RESO))
		frmEndIndx = int(np.floor(segEndT/FRM_RESO))+1
		
		# Get the note index to be tracked within the current segment
		note_indx_track = []
		for j,noteInd in enumerate(note_inds):
			if note_onsets[j]<segStartT and note_offsets[j]>segStartT:
				note_indx_track.append(noteInd)

		# Get the index of frequncy bins to be tracked
		freq_indx_track = []
		for j,noteInd in enumerate(note_indx_track):
			f0freq = f0[int(noteInd)]
			inhar = B1[int(noteInd)]
			nH = int(np.floor(((-1+(1+4*inhar*((SR/3/f0freq)**2))**0.5)/(2*inhar))**0.5))
			onsetT = note_onsets[j]
			offsetT = note_offsets[j]
			hfs = np.arange(1, nH+1)*f0freq*(1+inhar*np.arange(1, nH+1)**2)**0.5
			hfs_freqInds = hfs/FREQ_RESO
			for hfs_freqInd in hfs_freqInds:
				hfs_freqInd = int(np.floor(hfs_freqInd))
				freq_indx_track.append(hfs_freqInd)
		freq_indx_track = list(set(freq_indx_track))

		# Calculate the rms of specified freqency & time index
		if len(freq_indx_track) != 0:
			S_seg = S_nopartial[freq_indx_track,frmStartIndx:frmEndIndx]
			rmse_reg = librosa.feature.rmse(S=S_seg)[0]
			rmse_reg[np.isnan(rmse_reg)] = 0
			rmse_feature[frmStartIndx:frmEndIndx] = rmse_reg
	
	# get rms after medfilt        
	second_32 = 60/tempo/8
	second_32_frm = np.ceil(second_32/FRM_RESO)
	if second_32_frm%2 > 0 :
		medwindow = int(second_32_frm)
	else:
		medwindow = int(second_32_frm-1)
	rmse_feature_med = scipy.signal.medfilt(rmse_feature,medwindow)
	
	# get rms in log format
	rmse_feature_med_log = np.copy(rmse_feature_med)
	rmse_feature_med_log[rmse_feature_med_log<EPS]=EPS
	rmse_feature_med_log = np.log(rmse_feature_med_log)
	mostValue = mode(rmse_feature_med_log)[0][0]
	rmse_feature_med_log[rmse_feature_med_log<mostValue] = mostValue

	# get diff from log rms
	rmse_feature_med_log_diff = np.insert(np.diff(rmse_feature_med_log),0,0)
	rmse_feature_med_log_diff[rmse_feature_med_log_diff<0]=0

	return rmse_feature_med, rmse_feature_med_log, rmse_feature_med_log_diff

def main(args):

	music_name = os.path.basename(args.input_dir)
	parent_dir = os.path.dirname(args.input_dir)

	# input path
	audio_path = os.path.join(args.input_dir, "{}.wav".format(music_name))
	transcription_path = os.path.join(args.input_dir, "{}_transcription.npy".format(music_name))
	f0_path = os.path.join(parent_dir, "f0.csv")
	B1_path = os.path.join(parent_dir, "B1.csv")

	# output path
	features_path = os.path.join(args.save_dir, "{}_features.npz".format(music_name))

	# get transcription results
	note_events = np.load(transcription_path)
	note_onsets = note_events[:,0]
	# get f0 and B1
	f0 = np.genfromtxt(f0_path, delimiter=',')
	B1 = np.genfromtxt(B1_path, delimiter=',')

	# this is slow for long audio files
	print("Computing features...")
	x, sr = librosa.load(audio_path,sr=SR,mono=True)
	offTime = librosa.get_duration(y=x, sr=SR)
	frm_time = np.arange(0,offTime,FRM_RESO)
	# estimate most likely tempo
	onset_env = librosa.onset.onset_strength(x, sr=SR)
	tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=SR)[0]
	# HPSS
	D = librosa.stft(x)
	H, P = librosa.decompose.hpss(D)

	# define segment by note_onsets
	second_16 = 60/tempo/4
	seg_starts = []
	segEnd = note_onsets[0]
	for i, onset in enumerate(note_onsets):
		if onset >= segEnd:
			segStart = onset
			for t in note_onsets[i:]:
				if t-segStart > second_16:
					segEnd = t
					break
			seg_starts.append(segStart)
	seg_starts = np.asarray(seg_starts)

	# define the start and end indexes of frames
	frm_startindx = []
	frm_endindx = []
	for i,t in enumerate(seg_starts):
		segStartT = t
		if t != seg_starts[-1]:
			segEndT = seg_starts[i+1]
		else:
			segEndT = offTime
		frm_startindx.append(int(np.floor(segStartT/FRM_RESO)))
		frm_endindx.append(int(np.floor(segEndT/FRM_RESO)))

	# calculate basic features
	rmse_feature_med, rmse_feature_med_log, rmse_feature_med_log_diff=get_rmse(H, note_events, f0, B1, seg_starts, tempo, offTime)

	# get features for each segment 
	max_linear = np.zeros(len(frm_startindx))
	max_db = np.zeros(len(frm_startindx))
	peak_loc = np.zeros(len(frm_startindx))
	for i,startindx in enumerate(frm_startindx):
		endindx = frm_endindx[i]
		max_linear_seg = rmse_feature_med[startindx:endindx]
		max_db_seg = rmse_feature_med_log[startindx:endindx]
		# decide which peak location to be recorded
		diffrmse_seg = rmse_feature_med_log_diff[startindx:endindx]
		maxdiff2nd_ind = diffrmse_seg.argsort()[-2:][::-1] # index of first two biggest changes
		vdiff2nd = max_db_seg[maxdiff2nd_ind[1]]-max_db_seg[maxdiff2nd_ind[0]]
		if vdiff2nd > 2:
			peak_ind = maxdiff2nd_ind[1]
		else:
			if vdiff2nd == 0:
				peak_ind = 0
			else:
				peak_ind = maxdiff2nd_ind[0]
		
		max_linear[i] = np.max(max_linear_seg)
		max_db[i] = np.max(max_db_seg)
		peak_loc[i] = peak_ind

	np.savez(features_path, max_linear=max_linear, max_db=max_db, peak_loc=peak_loc)
	print("Done!")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Get features for legato-pedal onset detection")
	parser.add_argument("input_dir",
						type=str,
						default = "input/chopin",
						help="Path to folder for input files"
						"including audio, transcription result"
						"f0 and inharmonicity coefficient of 88 notes.")
	parser.add_argument("save_dir",
						type=str,
						default = "features",
						help="Path to folder for saving features")

	main(parser.parse_args())
