import os
import mysql.connector
import configNeomuleDB as config
import openslide
from tifffile import memmap
import subprocess
import sys
import getopt

def getOpenSlide(gID):
	#GET HALO JOB IDs and CLASSIFIER IDs FROM GRIDTOOL DB
	dsn = mysql.connector.connect(**config.gridtool_CONFIG)
    	cursor = dsn.cursor()
    	query = ''.join(["select haloAnalysisJobId, idAnalysis from classifier_accum_runs where gridId = '",
		str(gID),"'"])
    	cursor.execute(query)
    	row = cursor.fetchone()
        hjIDs = row[0]
	dsn.close()

	#GET IMAGE IDS FROM HALO DB
    	dsn = mysql.connector.connect(**config.halo_CONFIG)
    	cursor = dsn.cursor()
    	query = ''.join(['select ImageId from analysisjob where Id = ',str(hjIDs)])
    	cursor.execute(query)
    	row = cursor.fetchone()
        iID = int(row[0])
    	dsn.close()

	#GET IMAGE LOCATIONS FROM HALO DB
    	dsn = mysql.connector.connect(**config.halo_CONFIG)
    	cursor = dsn.cursor()
    	query = ''.join(['select Tag from image where Id = ',str(iID)])
    	cursor.execute(query)
    	row = cursor.fetchone()
        iLOC = str(row[0]).split('\\')[4:]
    	dsn.close()

	#CHECK IF SVS IS AVAILABLE OFFLINE AND DOWNLOAD
	net_file = '/'.join(['/media/nt','/'.join(iLOC)])
	loc_file = '/'.join(['/media/bigG/IMAGES',iLOC[-1]])
	if not os.path.exists(loc_file):
		print('downloading svs file')
		sh = " ".join(['cp',net_file,'/media/bigG/IMAGES/'])
		subprocess.call(sh, shell = True)
	print('file ready')
	print(loc_file)
	return openslide.OpenSlide(loc_file)

def getGrid(gID,f='f'):
	dsn = mysql.connector.connect(**config.gridtool_CONFIG)
	cursor = dsn.cursor()
	query = ''.join(["select * from locality_runs where gridId = '",str(gID),"'"])
	cursor.execute(query)
	rows = cursor.fetchone()
	print(rows[0])
	AID = rows[0]
	dsn.close()

	dsn = mysql.connector.connect(**config.gridtool_CONFIG)
	cursor = dsn.cursor()
	query = ''.join(["select * from locality_tiles where idAnalysis = '",str(AID),"'"])
	cursor.execute(query)
	CLST3 = []
	row = cursor.fetchone()
	while row is not None:
		if f!= 'f':
	   		CLST3.append([row[7],row[8],row[170]])
		if f == 'f':
			CLST3.append([row[7],row[8]])
    		row = cursor.fetchone()
	dsn.close()
	return CLST3

def getFibersMask(gID):
        svs = getOpenSlide(gID)
        fnam = str(svs.properties['aperio.Filename'])
        path = "".join(['./masks/',fnam,'_msk.tiff'])
        if os.path.exists(path):
                return memmap(path,'r')
        else:
                print('Could not find the mask.')
                return None

def getCollagenMask(gID):
        svs = getOpenSlide(gID)
        fnam = str(svs.properties['aperio.Filename'])
        path = "".join(['../data/masks/',fnam,'_msk.svs'])
        if os.path.exists(path):
                return memmap(path,dtype='uint8')
        else:
                print('Could not find the mask.')
                return None


def readGridIDs(argv):
        grids = []
        try:
                opts, args = getopt.getopt(argv,"hg:",["grids="])
        except getopt.GetoptError:
                print('test.py -g <gridID1> -g <gridID2>')
                sys.exit(2)
        for opt, arg in opts:
                if opt in ('-h','--help'):
                        print('test.py -g <gridID1> -g <gridID2>')
                        sys.exit()
                elif opt in ("-g", "--grids"):
                        grids.append(arg)

        return(grids)

def getLocalityDistances(gID):
	dsn = mysql.connector.connect(**config.gridtool_CONFIG)
	cursor = dsn.cursor()
	query = "select * from distance_runs where gridId = '"+str(gID)+"'"
	cursor.execute(query)
	rows = cursor.fetchone()
	AID = rows[0]
	query = "select * from distance_tiles where idAnalysis = '"+str(AID)+"'"
	cursor.execute(query)
	CLST3 = []
	row = cursor.fetchone()
	while row is not None:
		CLST3.append(row[10])
		row = cursor.fetchone()
	dsn.close()
	return CLST3

