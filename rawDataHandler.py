import sys
import os
from pathlib import Path
from itertools import chain
import pandas as pd

trainPath = Path('C-NMC_Leukemia/C-NMC_training_data/')

prelimPhaseLabels = Path('C-NMC_Leukemia/C-NMC_test_prelim_phase_data/C-NMC_test_prelim_phase_data_labels.csv')
prelimPhaseData = Path('C-NMC_Leukemia/C-NMC_test_prelim_phase_data/C-NMC_test_prelim_phase_data/')

finalPhaseData = Path('C-NMC_Leukemia/C-NMC_test_final_phase_data/C-NMC_test_final_phase_data/')


class Patient():
    def __init__(self, patientID):
        self.patientID = patientID
        self.listOfALLCells = []
        self.listOfHEMCells = []

    def addALLCell(self, imgPath):
        self.listOfALLCells.append(imgPath)

    def addHEMCell(self, imgPath):
        self.listOfHEMCells.append(imgPath)

    def ALLCellsCount(self):
        return len(self.listOfALLCells)

    def HEMCellsCount(self):
        return len(self.listOfHEMCells)


class Patients():
    def __init__(self):
        self.patients = {}
        self.rename_prelimPhaseData()
        self.loadImages()
        pass

    def rename_prelimPhaseData(self):
        labels = pd.read_csv(prelimPhaseLabels, index_col=0)
        for row in labels.iterrows():
            imgName, imgNumber = row[0], row[1][0]
            oldName = prelimPhaseData/imgNumber
            newName = prelimPhaseData/imgName
            if os.path.isfile(oldName):
                os.rename(oldName, newName)
                print(f'Rename {imgNumber} TO {imgName}')

    def loadImages(self):
        trainImgs = trainPath.glob('*/*/*.bmp')# fold_[0,1,2]/[all,hem]/*bmp
        prelimPhaseImgs = prelimPhaseData.glob('*.bmp')
        trainDataset = chain(trainImgs, prelimPhaseImgs)# 'C-NMC_Leukemia/C-NMC_training_data/' + 'C-NMC_Leukemia/C-NMC_test_prelim_phase_data/C-NMC_test_prelim_phase_data/'
        for imgPath in trainDataset:
            patientID = imgPath.stem.split('_')[1]
            cellType = imgPath.stem.split('_')[-1]
            if patientID not in self.patients:
                newPatient = Patient(patientID)
                self.patients[patientID] = newPatient
            if cellType=='all':
                self.patients[patientID].addALLCell(imgPath)
            else:
                self.patients[patientID].addHEMCell(imgPath)

    def totalCells(self, cellType):
        total = 0
        for _, patient in self.patients.items():
            if cellType=='ALL':
                total += patient.ALLCellsCount()
            elif cellType=='HEM':
                total += patient.HEMCellsCount()
            else:
                raise Exception('Especify cell type HEM or ALL')
        return total

    def totalPatients(self):
        return len(self.patients.keys())

    def getPatientsIDs(self):
        return list(self.patients.keys())


    def getCellsImgPath(self, cellType):
        ret = []
        for _, patient in self.patients.items():
            if cellType=='ALL':
                ret += patient.listOfALLCells
            elif cellType=='HEM':
                ret += patient.listOfHEMCells
            else:
                raise Exception('Especify cell type HEM or ALL')
        return ret

    def getPatientCellsPath(self, patientID):
        patient = self.patients[patientID]
        if patient.ALLCellsCount()==0:
            return patient.listOfHEMCells.copy()
        else:
            return patient.listOfALLCells.copy()

    def getIdsALLPatients(self):
        ret = []
        for pId, patient in self.patients.items():
            if patient.ALLCellsCount()>0:
                ret.append(pId)
        return ret

    def getIdsHEMPatients(self):
        ret = []
        for pId, patient in self.patients.items():
            if patient.HEMCellsCount()>0:
                ret.append(pId)
        return ret

patients = Patients()

if __name__ == '__main__':
    print(f"Total ALL:{patients.totalCells(cellType='ALL')}")
    print(f"Total HEM:{patients.totalCells(cellType='HEM')}")
    print(f"Total Patients:{patients.totalPatients()}")
    print(f"\n{'#'*50}\n")
    for patientID, patient in patients.patients.items():
        print(f"ID:{patientID}; ALL count:{patient.ALLCellsCount()}; HEM count:{patient.HEMCellsCount()} ")

    print(f"Patients ID's: {patients.getPatientsIDs()}\n")
    print(f"ALL - Patients ID's ({len(patients.getIdsALLPatients())} patients): {patients.getIdsALLPatients()}\n")
    print(f"HEM - Patients ID's ({len(patients.getIdsHEMPatients())} patients): {patients.getIdsHEMPatients()}\n")

    print(f"H28 cells: {patients.getPatientCellsPath('H28')}")

    print(f"\nEnd Script!\n{'#'*50}")
