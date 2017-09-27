#! /usr/bin/python3
# -*- coding: utf-8 -*-


'''
Builds samples of data offline. Sample size is the same as dev. 
First argument is the path to training files.
Second argument is the language pair. 
Third arugment is the path to where to write the samples.
Fourth argument is the number of samples per (training) file
ex: ./offlinesampling.py /home/staff/sharid/nobackup/DiscoDataPredictorFormat/ en-fr /home/staff7/sharid/offlinesampling/ 100

'''


import sys
import numpy

path = sys.argv[1]
pair = sys.argv[2]
write = sys.argv[3]
samples = int(sys.argv[4])

training_files = ["Europarl."+pair+".raw-features", "IWSLT15."+pair+".raw-features", "NCv9."+pair+".raw-features"] 
dev = open(path + "all_dev."+pair+".raw-features", "r", encoding = "utf-8")


def readData(file):
	'''(file)-> dict {ce:[3,4,5,6,7....], elle:[5,24.2, ...], ...}
	'''
	sentence = 0
	PronsSents = {}
	for line in file:	
		parts = line.strip().split('\t')
		if len(parts) == 2:  #sentence pair
			sentence += 1
		elif len(parts) == 1:  # pronoun instance
			pron = line.strip().split(" ")[0]
			if pron in PronsSents:
				PronsSents[pron].append(sentence)
			else:
				PronsSents[pron] = [sentence]
		else:
			print ("something wrong when reading the file ", line, len(parts))
			print ('sentences read ', sentence)
	return PronsSents



#get distribution of dev
devPronouns = readData(dev)
dev.close()

#dmcounts
devCounts = {}
for pron in devPronouns:
	devCounts[pron] = len(devPronouns[pron])

# sample size
sample_size = sum(devCounts.values())


print ("devCounts --->", devCounts)

def removeDuplicates(lst):
	dset = set()
	return [l for l in lst if l not in dset and not dset.add(l)]


def getOneSample(fil):
	all_sentences = []
	for pronoun in fil:
		selected = numpy.random.choice(numpy.asarray(trainPronouns[pronoun]), devCounts[pronoun], replace=True)
		sentences_in_sample = selected.tolist()
		all_sentences += sentences_in_sample
	nodoubles = removeDuplicates(all_sentences)
	return nodoubles



for j in range(len(training_files)):

	trainf = open(path + training_files[j], "r", encoding="utf-8")
	trainPronouns = readData(trainf)
	trainf.close()

	# samples per file
	for i in range(samples):
	
		oneSample = getOneSample(trainPronouns)
		
		#write sample file
		new = open(write + pair + "/" + "sample" + str(j)+ "-" + str(i), "w", encoding = "utf-8")
	
		#read again
		f = open(path + training_files[j], "r", encoding="utf-8")
		sentCounter = 0
		for line in f:
			parts = line.strip().split('\t')
			if len(parts) == 2: #sentence line
				sentCounter+=1
				if sentCounter in oneSample:
					new.write(line)
			if len(parts) == 1:  #pronoun line
				if sentCounter in oneSample:
					new.write(line)
		f.close()
		new.close()
		

	print ("end of sampling file--->", training_files[j])














	
