import os
import multiprocessing
import threading
import queue
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import radiomics
radiomics.setVerbosity(60)#Quiet
import SimpleITK as sitk
from scipy.spatial import distance
from scipy.signal import resample

augm_patLvDiv_train = Path('data/augm_patLvDiv_train/')
augm_patLvDiv_valid = Path('data/augm_patLvDiv_valid')
patLvDiv_test = Path('data/patLvDiv_test')

def readImage(path, color):
    if color=='rgb':
        return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    elif color=='gray':
        return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
    elif color=='hsv':
        return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2HSV)
    return None

def getPyRadImageAndMask(grayScaleImage, jointSeries=False):
    _,pyRadMask = cv2.threshold(grayScaleImage, 1, 1, cv2.THRESH_BINARY)
    pyRadMask = cv2.erode(pyRadMask, np.ones((2,2), np.uint8), iterations=1)
    pyRadimage = sitk.GetImageFromArray(grayScaleImage)
    pyRadMask = sitk.GetImageFromArray(pyRadMask)
    #Datails on 'jointSeries' in https://github.com/Radiomics/pyradiomics/issues/447
    if jointSeries:
        pyRadimage = sitk.JoinSeries(pyRadimage)
        pyRadMask = sitk.JoinSeries(pyRadMask)
    return pyRadimage, pyRadMask

def getFirstOrderFeatures(image):
    image, mask = getPyRadImageAndMask(image)
    rad = radiomics.firstorder.RadiomicsFirstOrder(image, mask)
    rad.execute()
    featuresDict = {}
    #Energy is a measure of the magnitude of voxel values in an image. A larger values implies a greater sum of the squares of these values.
    featuresDict['1stOrderFeats_energy'] = rad.getEnergyFeatureValue()
    #Total Energy is the value of Energy feature scaled by the volume of the voxel in cubic mm
    #featuresDict['total_Energy'] = rad.getTotalEnergyFeatureValue()
    #Entropy specifies the uncertainty/randomness in the image values. It measures the average amount of information required to encode the image values.
    featuresDict['1stOrderFeats_entropy'] = rad.getEntropyFeatureValue()
    #Minimum value
    featuresDict['1stOrderFeats_minimum_value'] = rad.getMinimumFeatureValue()
    #Maximum value
    featuresDict['1stOrderFeats_maximum_value'] = rad.getMaximumFeatureValue()
    #Range (Max - Min)
    featuresDict['1stOrderFeats_range'] = rad.getRangeFeatureValue()
    #90% of the data values lie below the 90th percentile
    featuresDict['1stOrderFeats_90th_percentile'] = rad.get90PercentileFeatureValue()
    #10% of the data values lie below the 10th percentile.
    featuresDict['1stOrderFeats_10th_percentile'] = rad.get10PercentileFeatureValue()
    #Difference between the 25th and 75th percentile of the image array
    featuresDict['1stOrderFeats_interquartile_range'] = rad.getInterquartileRangeFeatureValue()
    #The average gray level intensity within the ROI
    featuresDict['1stOrderFeats_mean'] = rad.getMeanFeatureValue()
    #Standard Deviation measures the amount of variation or dispersion from the Mean Value
    featuresDict['1stOrderFeats_standard_deviation'] = rad.getStandardDeviationFeatureValue()
    #Variance is the the mean of the squared distances of each intensity value from the Mean value. This is a measure of the spread of the distribution about the mean.
    featuresDict['1stOrderFeats_variance'] = rad.getVarianceFeatureValue()
    #The median gray level intensity within the ROI
    featuresDict['1stOrderFeats_median'] = rad.getMedianFeatureValue()
    #Mean Absolute Deviation is the mean distance of all intensity values from the Mean Value of the image array
    featuresDict['1stOrderFeats_mean_absolute_deviation'] = rad.getMeanAbsoluteDeviationFeatureValue()
    #Robust Mean Absolute Deviation is the mean distance of all intensity values from the Mean Value calculated on the subset of image array
    #with gray levels in between, or equal to the 10th and 90th percentile
    featuresDict['1stOrderFeats_robust_mean_absolute_deviation'] = rad.getRobustMeanAbsoluteDeviationFeatureValue()
    #RMS is the square-root of the mean of all the squared intensity values. It is another measure of the magnitude of the image values
    featuresDict['1stOrderFeats_root_mean_squared'] = rad.getRootMeanSquaredFeatureValue()
    #Skewness measures the asymmetry of the distribution of values about the Mean value.
    #Depending on where the tail is elongated and the mass of the distribution is concentrated, this value can be positive or negative
    featuresDict['1stOrderFeats_skewness'] = rad.getSkewnessFeatureValue()
    #Kurtosis is a measure of the ‘peakedness’ of the distribution of values in the image ROI.
    #A higher kurtosis implies that the mass of the distribution is concentrated towards the tail(s) rather than towards the mean.
    #A lower kurtosis implies the reverse: that the mass of the distribution is concentrated towards a spike near the Mean value
    featuresDict['1stOrderFeats_kurtosis'] = rad.getKurtosisFeatureValue()
    #Uniformity is a measure of the sum of the squares of each intensity value.
    #This is a measure of the homogeneity of the image array, where a greater uniformity implies a greater homogeneity or a smaller range of discrete intensity values.
    featuresDict['1stOrderFeats_uniformity'] = rad.getUniformityFeatureValue()
    return featuresDict

#############################################################################################################################################################################
#***************************************************************************************************************************************************************************#
#***************************************************************************************************************************************************************************#
#***************************************************************************************************************************************************************************#
#############################################################################################################################################################################


def getGLCMFeatures(image):
    image, mask = getPyRadImageAndMask(image, jointSeries=True)
    rad = radiomics.glcm.RadiomicsGLCM(image, mask)
    rad.execute()
    featuresDict = {}
    #Autocorrelation is a measure of the magnitude of the fineness and coarseness of texture
    featuresDict['GLCM_autocorrelation'] = rad.getAutocorrelationFeatureValue()
    #Returns the mean gray level intensity of the i distribution.
    featuresDict['GLCM_joint_average'] = rad.getJointAverageFeatureValue()
    #Cluster Prominence is a measure of the skewness and asymmetry of the GLCM.
    #A higher values implies more asymmetry about the mean while a lower value indicates a peak near the mean value and less variation about the mean
    featuresDict['GLCM_cluster_prominence'] = rad.getClusterProminenceFeatureValue()
    #Cluster Shade is a measure of the skewness and uniformity of the GLCM.
    #A higher cluster shade implies greater asymmetry about the mean.
    featuresDict['GLCM_cluster_shade'] = rad.getClusterShadeFeatureValue()
    #Cluster Tendency is a measure of groupings of voxels with similar gray-level values
    featuresDict['GLCM_cluster_tendency'] = rad.getClusterTendencyFeatureValue()
    #Contrast is a measure of the local intensity variation, favoring values away from the diagonal (i=j).
    #A larger value correlates with a greater disparity in intensity values among neighboring voxels.
    featuresDict['GLCM_contrast'] = rad.getContrastFeatureValue()
    #Correlation is a value between 0 (uncorrelated) and 1 (perfectly correlated) 
    #showing the linear dependency of gray level values to their respective voxels in the GLCM.
    featuresDict['GLCM_correlation'] = rad.getCorrelationFeatureValue()
    #Difference Average measures the relationship between occurrences of pairs with similar intensity values and occurrences of pairs with differing intensity values.
    featuresDict['GLCM_difference_average'] = rad.getDifferenceAverageFeatureValue()
    #Difference Entropy is a measure of the randomness/variability in neighborhood intensity value differences.
    featuresDict['GLCM_difference_entropy'] = rad.getDifferenceEntropyFeatureValue()
    #Difference Variance is a measure of heterogeneity that places higher weights on differing intensity level pairs that deviate more from the mean.
    featuresDict['GLCM_difference_variance'] = rad.getDifferenceVarianceFeatureValue()
    #Energy is a measure of homogeneous patterns in the image.
    #A greater Energy implies that there are more instances of intensity value pairs in the image that neighbor each other at higher frequencies.
    featuresDict['GLCM_joint_energy'] = rad.getJointEnergyFeatureValue()
    #Joint entropy is a measure of the randomness/variability in neighborhood intensity values
    featuresDict['GLCM_joint_entropy'] = rad.getJointEntropyFeatureValue()
    #IMC1 assesses the correlation between the probability distributions of i and j (quantifying the complexity of the texture), using mutual information I(x, y):
    featuresDict['GLCM_informational_measure_correlation_1'] = rad.getImc1FeatureValue()
    #IMC2 also assesses the correlation between the probability distributions of i and j (quantifying the complexity of the texture)
    featuresDict['GLCM_informational_measure_correlation_2'] = rad.getImc2FeatureValue()
    #IDM (a.k.a Homogeneity 2) is a measure of the local homogeneity of an image. IDM weights are the inverse of the Contrast weights
    featuresDict['GLCM_inverse_difference_moment'] = rad.getIdmFeatureValue()
    #The Maximal Correlation Coefficient is a measure of complexity of the texture and 0≤MCC≤1.
    featuresDict['GLCM_maximal_correlation_coefficient'] = rad.getMCCFeatureValue()
    #IDMN (inverse difference moment normalized) is a measure of the local homogeneity of an image.
    featuresDict['GLCM_inverse_difference_moment_normalized'] = rad.getIdmnFeatureValue()
    #ID (a.k.a. Homogeneity 1) is another measure of the local homogeneity of an image.
    #With more uniform gray levels, the denominator will remain low, resulting in a higher overall value
    featuresDict['GLCM_inverse_difference'] = rad.getIdFeatureValue()
    #DN (inverse difference normalized) is another measure of the local homogeneity of an image.
    #Unlike Homogeneity1, IDN normalizes the difference between the neighboring intensity values by dividing over the total number of discrete intensity values
    featuresDict['GLCM_inverse_difference_normalized'] = rad.getIdnFeatureValue()
    featuresDict['GLCM_inverse_variance'] = rad.getInverseVarianceFeatureValue()
    #Maximum Probability is occurrences of the most predominant pair of neighboring intensity values
    featuresDict['GLCM_maximum_probability'] = rad.getMaximumProbabilityFeatureValue()
    #Sum Average measures the relationship between occurrences of pairs with lower intensity values and occurrences of pairs with higher intensity values
    featuresDict['GLCM_sum_average'] = rad.getSumAverageFeatureValue()
    #Sum Entropy is a sum of neighborhood intensity value differences
    featuresDict['GLCM_sum_entropy'] = rad.getSumEntropyFeatureValue()
    #Sum of Squares or Variance is a measure in the distribution of neigboring intensity level pairs about the mean intensity level in the GLCM
    featuresDict['GLCM_sum_of_squares'] = rad.getSumSquaresFeatureValue()
    return featuresDict

def getGLDMFeatures(image):
    image, mask = getPyRadImageAndMask(image, jointSeries=True)
    rad = radiomics.gldm.RadiomicsGLDM(image, mask)
    rad.execute()
    featuresDict = {}
    #A measure of the distribution of small dependencies, with a greater value indicative of smaller dependence and less homogeneous textures.
    featuresDict['GLDM_small_dependence_emphasis'] = rad.getSmallDependenceEmphasisFeatureValue()
    #A measure of the distribution of large dependencies, with a greater value indicative of larger dependence and more homogeneous textures.
    featuresDict['GLDM_large_dependence_emphasis'] = rad.getLargeDependenceEmphasisFeatureValue()
    #Measures the similarity of gray-level intensity values in the image, where a lower GLN value correlates with a greater similarity in intensity values.
    featuresDict['GLDM_gray_level_nonUniformity'] = rad.getGrayLevelNonUniformityFeatureValue()
    #Measures the similarity of dependence throughout the image, with a lower value indicating more homogeneity among dependencies in the image.
    featuresDict['GLDM_dependence_nonUniformity'] = rad.getDependenceNonUniformityFeatureValue()
    #Measures the similarity of dependence throughout the image,
    #with a lower value indicating more homogeneity among dependencies in the image. This is the normalized version of the DLN formula.
    featuresDict['GLDM_dependence_nonUniformity_normalized'] = rad.getDependenceNonUniformityNormalizedFeatureValue()
    #Measures the variance in grey level in the image.
    featuresDict['GLDM_gray_level_variance'] = rad.getGrayLevelVarianceFeatureValue()
    #Measures the variance in dependence size in the image.
    featuresDict['GLDM_dependence_variance'] = rad.getDependenceVarianceFeatureValue()
    #Measures the variance in dependence size in the image.
    featuresDict['GLDM_dependence_entropy'] = rad.getDependenceEntropyFeatureValue()
    #Measures the distribution of low gray-level values, with a higher value indicating a greater concentration of low gray-level values in the image.
    featuresDict['GLDM_low_gray_level_emphasis'] = rad.getLowGrayLevelEmphasisFeatureValue()
    #Measures the distribution of the higher gray-level values, with a higher value indicating a greater concentration of high gray-level values in the image.
    featuresDict['GLDM_high_gray_level_emphasis'] = rad.getHighGrayLevelEmphasisFeatureValue()
    #Measures the joint distribution of small dependence with lower gray-level values.
    featuresDict['GLDM_small_dependence_low_gray_level_emphasis'] = rad.getSmallDependenceLowGrayLevelEmphasisFeatureValue()
    #Measures the joint distribution of small dependence with higher gray-level values.
    featuresDict['GLDM_small_dependence_high_gray_level_emphasis'] = rad.getSmallDependenceHighGrayLevelEmphasisFeatureValue()
    #Measures the joint distribution of large dependence with lower gray-level values.
    featuresDict['GLDM_large_dependence_low_gray_level_emphasis'] = rad.getLargeDependenceLowGrayLevelEmphasisFeatureValue()
    #Measures the joint distribution of large dependence with higher gray-level values.
    featuresDict['GLDM_large_dependence_high_gray_level_emphasis'] = rad.getLargeDependenceHighGrayLevelEmphasisFeatureValue()
    return featuresDict

def getGLRLMFeatures(image):
    image, mask = getPyRadImageAndMask(image, jointSeries=True)
    rad = radiomics.glrlm.RadiomicsGLRLM(image, mask)
    rad.execute()
    featuresDict = {}
    #SRE is a measure of the distribution of short run lengths, with a greater value indicative of shorter run lengths and more fine textural textures.
    featuresDict['GLRLM_short_run_emphasis'] = rad.getShortRunEmphasisFeatureValue()
    #LRE is a measure of the distribution of long run lengths, with a greater value indicative of longer run lengths and more coarse structural textures.
    featuresDict['GLRLM_long_run_emphasis'] = rad.getLongRunEmphasisFeatureValue()
    #GLN measures the similarity of gray-level intensity values in the image, where a lower GLN value correlates with a greater similarity in intensity values.
    featuresDict['GLRLM_gray_level_nonUniformity'] = rad.getGrayLevelNonUniformityFeatureValue()
    #GLNN measures the similarity of gray-level intensity values in the image,
    #where a lower GLNN value correlates with a greater similarity in intensity values. This is the normalized version of the GLN formula.
    featuresDict['GLRLM_gray_level_nonUniformity_normalized'] = rad.getGrayLevelNonUniformityNormalizedFeatureValue()
    #RLN measures the similarity of run lengths throughout the image,
    #with a lower value indicating more homogeneity among run lengths in the image.
    featuresDict['GLRLM_run_length_nonUniformity'] = rad.getRunLengthNonUniformityFeatureValue()
    #RLNN measures the similarity of run lengths throughout the image,
    #with a lower value indicating more homogeneity among run lengths in the image. This is the normalized version of the RLN formula.
    featuresDict['GLRLM_run_length_nonUniformity_normalized'] = rad.getRunLengthNonUniformityNormalizedFeatureValue()
    #RP measures the coarseness of the texture by taking the ratio of number of runs and number of voxels in the ROI.
    featuresDict['GLRLM_run_percentage'] = rad.getRunPercentageFeatureValue()
    #GLV measures the variance in gray level intensity for the runs.
    featuresDict['GLRLM_gray_level_variance'] = rad.getGrayLevelVarianceFeatureValue()
    #RV is a measure of the variance in runs for the run lengths.
    featuresDict['GLRLM_run_variance'] = rad.getRunVarianceFeatureValue()
    #RE measures the uncertainty/randomness in the distribution of run lengths and gray levels. A higher value indicates more heterogeneity in the texture patterns.
    featuresDict['GLRLM_run_entropy'] = rad.getRunEntropyFeatureValue()
    #LGLRE measures the distribution of low gray-level values, with a higher value indicating a greater concentration of low gray-level values in the image.
    featuresDict['GLRLM_low_gray_level_run_emphasis'] = rad.getLowGrayLevelRunEmphasisFeatureValue()
    #HGLRE measures the distribution of the higher gray-level values, with a higher value indicating a greater concentration of high gray-level values in the image.
    featuresDict['GLRLM_high_gray_level_run_emphasis'] = rad.getHighGrayLevelRunEmphasisFeatureValue()
    #SRLGLE measures the joint distribution of shorter run lengths with lower gray-level values.
    featuresDict['GLRLM_short_run_low_gray_level_emphasis'] = rad.getShortRunLowGrayLevelEmphasisFeatureValue()
    #SRHGLE measures the joint distribution of shorter run lengths with higher gray-level values.
    featuresDict['GLRLM_short_run_high_gray_level_emphasis'] = rad.getShortRunHighGrayLevelEmphasisFeatureValue()
    #LRLGLRE measures the joint distribution of long run lengths with lower gray-level values.
    featuresDict['GLRLM_long_run_low_gray_level_emphasis'] = rad.getLongRunLowGrayLevelEmphasisFeatureValue()
    #LRHGLRE measures the joint distribution of long run lengths with higher gray-level values.
    featuresDict['GLRLM_long_run_high_gray_level_emphasis'] = rad.getLongRunHighGrayLevelEmphasisFeatureValue()
    return featuresDict

def getGLSZMFeatures(image):
    image, mask = getPyRadImageAndMask(image, jointSeries=True)
    rad = radiomics.glszm.RadiomicsGLSZM(image, mask)
    rad.execute()
    featuresDict = {}
    #SAE is a measure of the distribution of small size zones, with a greater value indicative of more smaller size zones and more fine textures
    featuresDict['GLSZM_small_area_emphasis'] = rad.getSmallAreaEmphasisFeatureValue()
    #LAE is a measure of the distribution of large area size zones, with a greater value indicative of more larger size zones and more coarse textures.
    featuresDict['GLSZM_large_area_emphasis'] = rad.getLargeAreaEmphasisFeatureValue()
    #GLN measures the variability of gray-level intensity values in the image, with a lower value indicating more homogeneity in intensity values.
    featuresDict['GLSZM_gray_level_nonUniformity'] = rad.getGrayLevelNonUniformityFeatureValue()
    #GLNN measures the variability of gray-level intensity values in the image,
    #with a lower value indicating a greater similarity in intensity values. This is the normalized version of the GLN formula.
    featuresDict['GLSZM_gray_level_nonUniformity_normalized'] = rad.getGrayLevelNonUniformityNormalizedFeatureValue()
    #SZN measures the variability of size zone volumes in the image, with a lower value indicating more homogeneity in size zone volumes.
    featuresDict['GLSZM_size_zone_nonUniformity'] = rad.getSizeZoneNonUniformityFeatureValue()
    #SZNN measures the variability of size zone volumes throughout the image, with a lower value indicating more homogeneity among zone size volumes in the image.
    #This is the normalized version of the SZN formula.
    featuresDict['GLSZM_size_zone_nonUniformity_normalized'] = rad.getSizeZoneNonUniformityNormalizedFeatureValue()
    #ZP measures the coarseness of the texture by taking the ratio of number of zones and number of voxels in the ROI.
    featuresDict['GLSZM_zone_percentage'] = rad.getZonePercentageFeatureValue()
    #GLV measures the variance in gray level intensities for the zones.
    featuresDict['GLSZM_gray_level_variance'] = rad.getGrayLevelVarianceFeatureValue()
    #ZV measures the variance in zone size volumes for the zones.
    featuresDict['GLSZM_zone_variance'] = rad.getZoneVarianceFeatureValue()
    #ZE measures the uncertainty/randomness in the distribution of zone sizes and gray levels. A higher value indicates more heterogeneneity in the texture patterns.
    featuresDict['GLSZM_zone_entropy'] = rad.getZoneEntropyFeatureValue()
    #LGLZE measures the distribution of lower gray-level size zones, with a higher value indicating a greater proportion of lower gray-level values and size zones in the image.
    featuresDict['GLSZM_low_gray_level_zone_emphasis'] = rad.getLowGrayLevelZoneEmphasisFeatureValue()
    #HGLZE measures the distribution of the higher gray-level values, with a higher value indicating a greater proportion of higher gray-level values and size zones in the image.
    featuresDict['GLSZM_high_gray_level_zone_emphasis'] = rad.getHighGrayLevelZoneEmphasisFeatureValue()
    #SALGLE measures the proportion in the image of the joint distribution of smaller size zones with lower gray-level values.
    featuresDict['GLSZM_small_area_low_gray_level_emphasis'] = rad.getSmallAreaLowGrayLevelEmphasisFeatureValue()
    #SAHGLE measures the proportion in the image of the joint distribution of smaller size zones with higher gray-level values.
    featuresDict['GLSZM_small_area_high_gray_level_emphasis'] = rad.getSmallAreaHighGrayLevelEmphasisFeatureValue()
    #LALGLE measures the proportion in the image of the joint distribution of larger size zones with lower gray-level values.
    featuresDict['GLSZM_large_area_low_gray_level_emphasis'] = rad.getLargeAreaLowGrayLevelEmphasisFeatureValue()
    #LAHGLE measures the proportion in the image of the joint distribution of larger size zones with higher gray-level values.
    featuresDict['GLSZM_large_area_high_gray_level_emphasis'] = rad.getLargeAreaHighGrayLevelEmphasisFeatureValue()
    return featuresDict

def getNGTDMFeatures(image):
    image, mask = getPyRadImageAndMask(image, jointSeries=True)
    rad = radiomics.ngtdm.RadiomicsNGTDM(image, mask)
    rad.execute()
    featuresDict = {}
    #Coarseness is a measure of average difference between the center voxel and its neighbourhood
    #and is an indication of the spatial rate of change. A higher value indicates a lower spatial change rate and a locally more uniform texture.
    featuresDict['NGTDM_coarseness'] = rad.getCoarsenessFeatureValue()
    #Contrast is a measure of the spatial intensity change, but is also dependent on the overall gray level dynamic range.
    #Contrast is high when both the dynamic range and the spatial change rate are high, i.e.
    #an image with a large range of gray levels, with large changes between voxels and their neighbourhood.
    featuresDict['NGTDM_contrast'] = rad.getContrastFeatureValue()
    #A measure of the change from a pixel to its neighbour.
    #A high value for busyness indicates a ‘busy’ image, with rapid changes of intensity between pixels and its neighbourhood.
    featuresDict['NGTDM_busyness'] = rad.getBusynessFeatureValue()
    #An image is considered complex when there are many primitive components in the image, i.e.
    #the image is non-uniform and there are many rapid changes in gray level intensity.
    featuresDict['NGTDM_complexity'] = rad.getComplexityFeatureValue()
    #Strenght is a measure of the primitives in an image.
    #Its value is high when the primitives are easily defined and visible, i.e.
    #an image with slow change in intensity but more large coarse differences in gray level intensities.
    featuresDict['NGTDM_strength'] = rad.getStrengthFeatureValue()
    return featuresDict

#############################################################################################################################################################################
#***************************************************************************************************************************************************************************#
#***************************************************************************************************************************************************************************#
#***************************************************************************************************************************************************************************#
#############################################################################################################################################################################

def getNucleusArea(grayScaleImage):
    _, thresh = cv2.threshold(grayScaleImage, 1, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, np.ones((2,2), np.uint8), iterations=1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    return cv2.contourArea(contour)

def getNucleusPerimeter(grayScaleImage):
    _, thresh = cv2.threshold(grayScaleImage, 1, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, np.ones((2,2), np.uint8), iterations=1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    return cv2.arcLength(contour, True)

def getConvexArea(grayScaleImage):
    _, thresh = cv2.threshold(grayScaleImage, 1, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, np.ones((2,2), np.uint8), iterations=1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour, returnPoints=True)
    return cv2.contourArea(hull)

def getConvexPerimeter(grayScaleImage):
    _,thresh = cv2.threshold(grayScaleImage, 1, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, np.ones((2,2), np.uint8), iterations=1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour, returnPoints=True)
    return cv2.arcLength(hull, True)

#DOI 10.1002/jemt.22718
def getEquivalentDiameter(grayScaleImage):
    return np.sqrt(4.0 * (getNucleusArea(grayScaleImage)/np.pi))

#DOI 10.1016/j.artmed.2014.09.002 **** DOI 10.1002/jemt.22718
def getCompactness(grayScaleImage):
    return (4.0 * np.pi * getNucleusArea(grayScaleImage)) / np.power(getNucleusPerimeter(grayScaleImage), 2)

#DOI 10.1016/j.artmed.2014.09.002 **** DOI 10.1002/jemt.22718
def getConvexity(grayScaleImage):
    return getConvexPerimeter(grayScaleImage) / getNucleusPerimeter(grayScaleImage)

#DOI 10.1016/j.artmed.2014.09.002 **** DOI 10.1002/jemt.22718
def getSolidity(grayScaleImage):
    return getNucleusArea(grayScaleImage) / getConvexArea(grayScaleImage)

#DOI 10.1016/j.artmed.2014.09.002 **** DOI 10.1002/jemt.22718
def getNucleosEccentricity(grayScaleImage):
    _, thresh = cv2.threshold(grayScaleImage, 1, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, np.ones((2,2), np.uint8), iterations=1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    (x, y), (minorAxis, majorAxis), _ = cv2.fitEllipse(contour)
    a = majorAxis/2
    b = minorAxis/2
    c = np.sqrt(a**2 - b**2)
    eccentricity = c/a
    aspectRatio = minorAxis/majorAxis
    return eccentricity, aspectRatio, majorAxis, minorAxis

#DOI 10.1016/j.artmed.2014.09.002
def getNucleusElongation(grayScaleImage):
    _, _, majorAxis, minorAxis = getNucleosEccentricity(grayScaleImage)
    return 1.0 - (minorAxis/majorAxis)

#DOI 10.1016/j.artmed.2014.09.002
def getRectangularity(grayScaleImage):
    area = getNucleusArea(grayScaleImage)
    _, _, majorAxis, minorAxis = getNucleosEccentricity(grayScaleImage)
    return area/(majorAxis*minorAxis)

#############################################################################################################################################################################
#***************************************************************************************************************************************************************************#
#***************************************************************************************************************************************************************************#
#***************************************************************************************************************************************************************************#
#############################################################################################################################################################################

def getContourSignature(grayScaleImage, sizeVector=320):
    _,thresh = cv2.threshold(grayScaleImage, 1, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, np.ones((2,2),np.uint8),iterations = 1)
    contours, _ = cv2.findContours(thresh, 1, 2)
    contour = max(contours, key=cv2.contourArea)
    m = cv2.moments(thresh)
    centroid = (int(m['m10']/m['m00']), int(m['m01']/m['m00']))
    signature = []
    for points in contour:
        curvePoint = points[0][0], points[0][1]
        signature.append(distance.euclidean(centroid, curvePoint))
    if (len(signature)>sizeVector) or (len(signature)<sizeVector):
        signature = resample(signature, sizeVector)
    signature = list(map(abs, np.fft.fft(signature)))
    signature = np.add(signature[0:len(signature)//2], 1e-9) #Add small constant (1e-9) to avoid 'nan' when np.log10(0)
    signature = np.log10(signature)
    return signature

#############################################################################################################################################################################
#***************************************************************************************************************************************************************************#
#***************************************************************************************************************************************************************************#
#***************************************************************************************************************************************************************************#
#############################################################################################################################################################################

def computeDCTransform(imgPath, noOfFeatures=1024):
    srcImg = readImage(imgPath, color='gray')
    #Compute DCT transform
    srcImg = np.array(srcImg, dtype=np.float32)
    dctFeats = cv2.dct(srcImg)
    #Thank's to https://stackoverflow.com/questions/50445847/how-to-zigzag-order-and-concatenate-the-value-every-line-using-python
    zigzagPattern = np.concatenate([np.diagonal(dctFeats[::-1,:], i)[::(2*(i % 2)-1)] for i in range(1-dctFeats.shape[0], dctFeats.shape[0])])
    zigzagPattern = zigzagPattern[0:noOfFeatures]
    zigzagPattern = np.log10(np.abs(zigzagPattern))
    return zigzagPattern

#############################################################################################################################################################################
#***************************************************************************************************************************************************************************#
#***************************************************************************************************************************************************************************#
#***************************************************************************************************************************************************************************#
#############################################################################################################################################################################

def extract_DCT_FeatureDict(imagePath):
    features = {}
    if 'all.bmp' in imagePath.name:
        features['cellType(ALL=1, HEM=-1)'] = 1
    elif 'hem.bmp' in imagePath.name:
        features['cellType(ALL=1, HEM=-1)'] = -1
    else:
        features['cellType(ALL=1, HEM=-1)'] = 0
    feats = computeDCTransform(imagePath)
    for i in range(len(feats)):
        features[f'DCT_pt{str(i).zfill(4)}'] = feats[i]
    return features

def extract_CONTOUR_FeatureDict(imagePath):
    features = {}
    if 'all.bmp' in imagePath.name:
        features['cellType(ALL=1, HEM=-1)'] = 1
    elif 'hem.bmp' in imagePath.name:
        features['cellType(ALL=1, HEM=-1)'] = -1
    else:
        features['cellType(ALL=1, HEM=-1)'] = 0
    img = cv2.cvtColor(cv2.imread(str(imagePath)), cv2.COLOR_BGR2GRAY)
    signature = getContourSignature(img)
    for i in range(len(signature)):
        features[f'CONTOUR_pt{str(i).zfill(3)}'] = signature[i]
    return features

def extract_MORPH_FeatureDict(imagePath):
    features = {}
    if 'all.bmp' in imagePath.name:
        features['cellType(ALL=1, HEM=-1)'] = 1
    elif 'hem.bmp' in imagePath.name:
        features['cellType(ALL=1, HEM=-1)'] = -1
    else:
        features['cellType(ALL=1, HEM=-1)'] = 0

    srcImg = cv2.imread(str(imagePath))
    grayImg = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)
    h, s, v = cv2.split(cv2.cvtColor(srcImg, cv2.COLOR_BGR2HSV))
    b, g, r = cv2.split(srcImg)
    #Color Features
    features['morphFeats_meanHue'] = np.mean(h)
    features['morphFeats_meanSaturation'] = np.mean(s)
    features['morphFeats_meanValue'] = np.mean(v)
    features['morphFeats_meanRed'] = np.mean(r)
    features['morphFeats_meanGreen'] = np.mean(g)
    features['morphFeats_meanBlue'] = np.mean(b)
    #Shape Features
    features['morphFeats_nucleoArea'] = getNucleusArea(grayImg)
    features['morphFeats_nucleoPerimeter'] = getNucleusPerimeter(grayImg)
    features['morphFeats_convexArea'] = getConvexArea(grayImg)
    features['morphFeats_convexPerimeter'] = getConvexPerimeter(grayImg)
    features['morphFeats_equivalentDiameter'] = getEquivalentDiameter(grayImg)
    features['morphFeats_compactness'] = getCompactness(grayImg)
    features['morphFeats_convexity'] = getConvexity(grayImg)
    features['morphFeats_solidity'] = getSolidity(grayImg)
    #Shape Features (Ellipse)
    eccentricity, aspectRatio, majorAxis, minorAxis = getNucleosEccentricity(grayImg)
    features['morphFeats_nucleoEccentricity'] = eccentricity
    features['morphFeats_nucleoAspectRatio'] = aspectRatio
    features['morphFeats_nucleoMajorAxis'] = majorAxis
    features['morphFeats_nucleoMinorAxis'] = minorAxis
    features['morphFeats_nucleoElongation'] = getNucleusElongation(grayImg)
    features['morphFeats_rectangularity'] = getRectangularity(grayImg)
    
    return features

def extract_TEXT_FeatureDict(imgPath):
    featuresDict = {}
    if 'all.bmp' in imgPath.name:
        featuresDict['cellType(ALL=1, HEM=-1)'] = 1
    elif 'hem.bmp' in imgPath.name:
        featuresDict['cellType(ALL=1, HEM=-1)'] = -1
    else:
        featuresDict['cellType(ALL=1, HEM=-1)'] = 0
    img = readImage(imgPath,'gray')

    height, width = img.shape
    #img45 = cv2.warpAffine(img, cv2.getRotationMatrix2D((width/2,height/2), 45, 1), (width, height))
    #img90 = cv2.warpAffine(img, cv2.getRotationMatrix2D((width/2,height/2), 90, 1), (width, height))
    #img135 = cv2.warpAffine(img, cv2.getRotationMatrix2D((width/2,height/2), 135, 1), (width, height))

    features0 = {**getGLCMFeatures(img), **getGLDMFeatures(img),
                 **getGLRLMFeatures(img), **getGLSZMFeatures(img),
                 **getNGTDMFeatures(img)}
    #features45 = {**getGLCMFeatures(img45), **getGLDMFeatures(img45),
    #              **getGLRLMFeatures(img45), **getGLSZMFeatures(img45),
    #              **getNGTDMFeatures(img45)}
    #features90 = {**getGLCMFeatures(img90), **getGLDMFeatures(img90),
    #              **getGLRLMFeatures(img90), **getGLSZMFeatures(img90),
    #              **getNGTDMFeatures(img90)}
    #features135 = {**getGLCMFeatures(img135), **getGLDMFeatures(img135),
    #               **getGLRLMFeatures(img135), **getGLSZMFeatures(img135),
    #               **getNGTDMFeatures(img135)}
    
    featNames = list(features0.keys())
    for i in range(len(features0)):
        featuresDict[f'Texture_0deg_{featNames[i]}'] = features0[featNames[i]]
    #    featuresDict[f'Texture_45deg_{featNames[i]}'] = features45[featNames[i]]
    #    featuresDict[f'Texture_90deg_{featNames[i]}'] = features90[featNames[i]]
    #    featuresDict[f'Texture_135deg_{featNames[i]}'] = features135[featNames[i]]
    return featuresDict

def extract_FOF_FeatureDict(imgPath):
    featuresDict = {}
    if 'all.bmp' in imgPath.name:
        featuresDict['cellType(ALL=1, HEM=-1)'] = 1
    elif 'hem.bmp' in imgPath.name:
        featuresDict['cellType(ALL=1, HEM=-1)'] = -1
    else:
        featuresDict['cellType(ALL=1, HEM=-1)'] = 0
    img = readImage(imgPath,'rgb')
    r, g, b = cv2.split(img)
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    feats0 = getFirstOrderFeatures(r)
    feats0 = {f'R_Chan_{k}': v for k, v in feats0.items()}
    feats1 = getFirstOrderFeatures(g)
    feats1 = {f'G_Chan_{k}': v for k, v in feats1.items()}
    feats2 = getFirstOrderFeatures(b)
    feats2 = {f'B_Chan_{k}': v for k, v in feats2.items()}
    feats3 = getFirstOrderFeatures(h)
    feats3 = {f'H_Chan_{k}': v for k, v in feats3.items()}
    feats4 = getFirstOrderFeatures(s)
    feats4 = {f'S_Chan_{k}': v for k, v in feats4.items()}
    feats5 = getFirstOrderFeatures(v)
    feats5 = {f'V_Chan_{k}': v for k, v in feats5.items()}
    featuresDict = {**featuresDict, **feats0, **feats1, **feats2, **feats3, **feats4, **feats5}
    return featuresDict

def extractFeatureDict(targetImgPath):
    fofFeats = extract_FOF_FeatureDict(targetImgPath)
    textFeats = extract_TEXT_FeatureDict(targetImgPath)
    morphFeats = extract_MORPH_FeatureDict(targetImgPath)
    contourFeats = extract_CONTOUR_FeatureDict(targetImgPath)
    dctFeats = extract_DCT_FeatureDict(targetImgPath)

    featuresDict = {}
    cellType = fofFeats.pop('cellType(ALL=1, HEM=-1)')
    textFeats.pop('cellType(ALL=1, HEM=-1)')
    morphFeats.pop('cellType(ALL=1, HEM=-1)')
    contourFeats.pop('cellType(ALL=1, HEM=-1)')
    dctFeats.pop('cellType(ALL=1, HEM=-1)')

    featuresDict['cellType(ALL=1, HEM=-1)'] = cellType
    return {**featuresDict, **fofFeats, **textFeats, **morphFeats, **contourFeats, **dctFeats}


#############################################################################################################################################################################
#***************************************************************************************************************************************************************************#
#***************************************************************************************************************************************************************************#
#***************************************************************************************************************************************************************************#
#############################################################################################################################################################################


def createAugmPatLvDivDataframe():
    TRAIN_IMGS = os.listdir(augm_patLvDiv_train)
    VALID_IMGS = os.listdir(augm_patLvDiv_valid)
    TEST_IMGS = os.listdir(patLvDiv_test)
    #Create Train Dataframe
    def createTrainDataframe():
        train_df = pd.DataFrame()
        np.random.shuffle(TRAIN_IMGS)
        for n, imgFile in enumerate(TRAIN_IMGS):
            featuresDict = {}
            imgPath = augm_patLvDiv_train / imgFile
            featuresDict = extractFeatureDict(imgPath)
            train_df = train_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Train')
        cols = train_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        train_df = train_df[cols]
        train_df.to_csv(f'feature-dataframes/AugmPatLvDiv_TRAIN-AllFeats_{train_df.shape[1]-1}-Features_{train_df.shape[0]}-images.csv')

    #Process for Validation dataset
    def createValidDataframe():
        valid_df = pd.DataFrame()
        np.random.shuffle(VALID_IMGS)
        for n, imgFile in enumerate(VALID_IMGS):
            featuresDict = {}
            imgPath = augm_patLvDiv_valid / imgFile
            featuresDict = extractFeatureDict(imgPath)
            valid_df = valid_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Validation')
        cols = valid_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        valid_df = valid_df[cols]
        valid_df.to_csv(f'feature-dataframes/AugmPatLvDiv_VALIDATION-AllFeats_{valid_df.shape[1]-1}-Features_{valid_df.shape[0]}-images.csv')

    #Process for Test dataset
    def createTestDataframe():
        test_df = pd.DataFrame()
        np.random.shuffle(TEST_IMGS)
        for n, imgFile in enumerate(TEST_IMGS):
            featuresDict = {}
            imgPath = patLvDiv_test / imgFile
            featuresDict = extractFeatureDict(imgPath)
            test_df = test_df.append(featuresDict, ignore_index=True)
            print(f'{n}-Test')
        cols = test_df.columns.tolist()
        cols.remove('cellType(ALL=1, HEM=-1)')
        cols = ['cellType(ALL=1, HEM=-1)'] + cols
        test_df = test_df[cols]
        test_df.to_csv(f'feature-dataframes/PatLvDiv_TEST-AllFeats_{test_df.shape[1]-1}-Features_{test_df.shape[0]}-images.csv')
    
    p0 = multiprocessing.Process(name='train_AugmPatLvDiv', target=createTrainDataframe)
    #p1 = multiprocessing.Process(name='valid_AugmPatLvDiv',target=createValidDataframe)
    #p2 = multiprocessing.Process(name='test_PatLvDiv',target=createTestDataframe)
    p0.start()
    #p1.start()
    #p2.start()
    p0.join()
    #p1.join()
    #p2.join()



if __name__ == '__main__':
    p0 = multiprocessing.Process(name='AugmPatLvDiv', target=createAugmPatLvDivDataframe)
    p0.start()
    p0.join()
    print(f"\nEnd Script!\n{'#'*50}")

















    """

    from matplotlib import pyplot as plt
    imgPath = augm_patLvDiv_train / 'AugmentedImg_4055_UID_H2_6_2_hem.bmp'
    img = readImage(imgPath,color='gray')
    getContourSignature(img, sizeVector=320)
    print(f"\nEnd Script!\n{'#'*50}")



    #img = readImage(Path('data/augm_patLvDiv_train/AugmentedImg_1001_UID_H23_10_1_hem.bmp'), color='gray')
    #img = readImage(Path('data/augm_patLvDiv_train/AugmentedImg_1002_UID_H14_6_3_hem.bmp'), color='gray')
    #img = readImage(Path('data/augm_patLvDiv_train/AugmentedImg_1003_UID_45_25_3_all.bmp'), color='gray')
    #img = readImage(Path('data/augm_patLvDiv_train/AugmentedImg_1003_UID_H12_15_7_hem.bmp'), color='gray')
    #img = readImage(Path('data/augm_patLvDiv_train/AugmentedImg_1003_UID_H22_17_7_hem.bmp'), color='gray')
    #img = readImage(Path('data/augm_patLvDiv_train/AugmentedImg_1003_UID_H22_28_1_hem.bmp'), color='gray')
    img = readImage(Path('data/augm_patLvDiv_train/AugmentedImg_1005_UID_H10_129_1_hem.bmp'), color='gray')

    from matplotlib import pyplot as plt
    #plt.imshow(img, cmap='gray')
    #plt.show()
    #getContourSignature(img, sizeVector=320)
    
    
    #Compute DCT transform
    img = np.array(img, dtype=np.float32)
    dctFeats = cv2.dct(img)
    #Thank's to https://stackoverflow.com/questions/50445847/how-to-zigzag-order-and-concatenate-the-value-every-line-using-python
    zigzagPattern = np.concatenate([np.diagonal(dctFeats[::-1,:], i)[::(2*(i % 2)-1)] for i in range(1-dctFeats.shape[0], dctFeats.shape[0])])
    zigzagPattern = zigzagPattern[0:1024]
    
    zigzagPattern = zigzagPattern.tolist()

    plt.subplot(311)
    plt.plot(zigzagPattern)
    plt.title('No_Log-Normalized dct', fontname='Times New Roman', fontsize=18)

    zigzagPattern = list(map(abs, zigzagPattern))

    plt.subplot(312)
    plt.plot(np.log(zigzagPattern))
    plt.title('Normal_Log-Normalized dct', fontname='Times New Roman', fontsize=18)

    plt.subplot(313)
    plt.plot(np.log10(zigzagPattern))
    plt.title('Log_10-Normalized dct', fontname='Times New Roman', fontsize=18)

    plt.show()
    """
