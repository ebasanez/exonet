import os
import sys
import re

class KOIFeatureExtractor:	
	
	COLUMN_ID = 0
	COLUMN_NAME = 1
	COLUMN_LABEL = 3
	COLUMN_PERIOD = 10
	COLUMN_T0 = 13
	COLUMN_DURATION = 19
	
	LABEL_TRUE = 'CONFIRMED'
	LABEL_FALSE = 'FALSE POSITIVE'
	TRAIN_LABELS = [LABEL_TRUE, LABEL_FALSE]
	LABEL_CANDIDATE ='CANDIDATE'
	
	def __init__(self, sourceFileName):
		self.sourceFileName = sourceFileName
 	
	def extractTrainData(self, destinationFileName):
		trainFileName = destinationFileName+"_train.csv"
		unlabeledFileName = destinationFileName+"_test.csv"
		
		print(f"Extracting features in dataset {self.sourceFileName} to files {trainFileName} and {unlabeledFileName}")
		 # Delete destination files to be able to recreate it
		self.deleteFile(trainFileName)
		self.deleteFile(unlabeledFileName) 
		with open(trainFileName,'a+') as trainFile,open(unlabeledFileName,'a+') as unlabeledFile:
			# Header row for generated file
			trainFile.write("mission,koi_id,koi_name,koi_time0bk,koi_period,koi_duration,koi_is_planet\n") 
			unlabeledFile.write("mission,koi_id,koi_name,koi_time0bk,koi_period,koi_duration\n") 
			with open(self.sourceFileName,'r') as sourceFile:
				# Skip header row
				next(sourceFile) 
				# Read each line in dataset
				for line in sourceFile:
					lineItems = line.split(',')
					koiId = lineItems[self.COLUMN_ID]
					koiName = lineItems[self.COLUMN_NAME]
					koiLabel = lineItems[self.COLUMN_LABEL]
					koiPeriod = float(lineItems[self.COLUMN_PERIOD])
					koiT0 = float(lineItems[self.COLUMN_T0])
					koiDuration = float(lineItems[self.COLUMN_DURATION])/24
					# We are only interested in KOIs with CONFIRMED of FALSE POSITIVE TCEs
					if koiLabel in self.TRAIN_LABELS: 
						trainFile.write('Kepler,%s,%s,%f,%f,%f,%d\n' % (koiId,koiName,koiT0,koiPeriod,koiDuration,self.toDummy(koiLabel)))
					if koiLabel == self.LABEL_CANDIDATE: 
						unlabeledFile.write('Kepler,%s,%s,%f,%f,%f\n' % (koiId,koiName,koiT0,koiPeriod,koiDuration))
					
	def toDummy(self, label):
		if label == self.LABEL_TRUE:
			return 1
		if label == self.LABEL_FALSE:
			return 0
		print("Error: label ", label, "not recognized")	
		return -1			
	
	def deleteFile(self, fileName):
		if os.path.exists(fileName):
			os.remove(fileName)
		
# Execute only if script run standalone (not imported)						
if __name__ == '__main__':
	(script, sourceFileName, trainFileName) = sys.argv
	extractor = KOIFeatureExtractor(sourceFileName)
	extractor.extractTrainData(trainFileName)