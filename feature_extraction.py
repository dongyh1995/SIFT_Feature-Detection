#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from scipy import ndimage
from scipy import misc
from scipy.stats import multivariate_normal
from scipy import signal
from numpy.linalg import norm 
import numpy.linalg
# def createDoG(img1, img2):
# 	DoG1 = 

def detect_features(imgname, threshold):
	# SIFT detector

	normal_image = ndimage.imread(imgname, flatten = True)

	# SIFT parametre
	s = 3
	k = 2 ** (1.0 / s)

	# Standard deviation for gaussian smoothing
	param1 = np.array([1.3, 1.6, 1.6 * k, 1.6 * (k ** 2), 1.6 * (k ** 3), 1.6 * (k ** 4)])
	param2 = np.array([1.6 * (k ** 2), 1.6 * (k ** 3), 1.6 * (k ** 4), 1.6 * (k ** 5), 1.6 * (k ** 6), 1.6 * (k ** 7)])
	param3 = np.array([1.6 * (k ** 5), 1.6 * (k ** 6), 1.6 * (k ** 7), 1.6 * (k ** 8), 1.6 * (k ** 9), 1.6 * (k ** 10)])
	param4 = np.array([1.6 * (k ** 8), 1.6 * (k ** 9), 1.6 * (k ** 10), 1.6 * (k ** 11), 1.6 * (k ** 12), 1.6 * (k ** 13)])
	paramtotal = np.array([1.6, 1.6 * k, 1.6 * (k ** 2), 1.6 * (k ** 3), 1.6 * (k ** 4), 1.6 * (k ** 5), 1.6 * (k ** 6), 1.6 * (k ** 7), 1.6 * (k ** 8), 1.6 * (k ** 9), 1.6 * (k ** 10), 1.6 * (k ** 11)])

	doubled_image = misc.imresize(normal_image, 200, 'bilinear').astype(int)
	normal_image = misc.imresize(doubled_image, 50, 'bilinear').astype(int)
	halved_image = misc.imresize(normal_image, 50, 'bilinear').astype(int)
	quarted_image = misc.imresize(halved_image,50, 'bilinear').astype(int)

	# Pyramid initilization
	pyrlv1 = np.zeros((doubled_image.shape[0], doubled_image.shape[1], 6))
	pyrlv2 = np.zeros((normal_image.shape[0], normal_image.shape[1], 6))
	pyrlv3 = np.zeros((halved_image.shape[0], halved_image.shape[1], 6))
	pyrlv4 = np.zeros((quarted_image.shape[0], quarted_image.shape[1], 6))

	# Gaussian smoothing and downsampling
	print "Constructing pyramid... "
	for i in range(6):
		pyrlv1[:, :, i] = ndimage.filters.gaussian_filter(doubled_image, param1[i])
		pyrlv2[:, :, i] = misc.imresize(ndimage.filters.gaussian_filter(doubled_image, param2[i]), 50, 'bilinear')
		pyrlv3[:, :, i] = misc.imresize(ndimage.filters.gaussian_filter(doubled_image, param3[i]), 25, 'bilinear')
		pyrlv4[:, :, i] = misc.imresize(ndimage.filters.gaussian_filter(doubled_image, param4[i]), 1.0 / 8.0, 'bilinear')

	# DoG pyramid initilization
	diff_pyrlv1 = np.zeros((doubled_image.shape[0], doubled_image.shape[1], 5))
	diff_pyrlv2 = np.zeros((normal_image.shape[0], normal_image.shape[1], 5))
	diff_pyrlv3 = np.zeros((halved_image.shape[0], halved_image.shape[1], 5))
	diff_pyrlv4 = np.zeros((quarted_image.shape[0], quarted_image.shape[1], 5))

	# Construct DoG Pyramid
	for i in range(5):
		diff_pyrlv1[:, :, i] = pyrlv1[:, :, i + 1] - pyrlv1[:, :, i]
		diff_pyrlv2[:, :, i] = pyrlv2[:, :, i + 1] - pyrlv2[:, :, i]
		diff_pyrlv3[:, :, i] = pyrlv3[:, :, i + 1] - pyrlv3[:, :, i]
		diff_pyrlv4[:, :, i] = pyrlv4[:, :, i + 1] - pyrlv4[:, :, i]


	extrPointLv1 = np.zeros((doubled_image.shape[0], doubled_image.shape[1], 3))
	extrPointLv2 = np.zeros((normal_image.shape[0], normal_image.shape[1], 3))
	extrPointLv3 = np.zeros((halved_image.shape[0], halved_image.shape[1], 3))
	extrPointLv4 = np.zeros((quarted_image.shape[0], quarted_image.shape[1], 3))

	print "Starting extreme point detection"

	print "First Octave"

	for i in range(1, 4):
		for j in range(80, doubled_image.shape[0] - 80):
			for k in range(80, doubled_image.shape[1] - 80):
				if(np.absolute(diff_pyrlv1[j, k, i]) < threshold):
					continue

				max_bool = diff_pyrlv1[j, k, i] > 0
				min_bool = diff_pyrlv1[j, k, i] < 0

				for di in range(-1, 2):
					for dj in range(-1, 2):
						for dk in range(-1, 2):
							if di == 0 and dj == 0  and dk == 0:
								continue

							max_bool = max_bool and diff_pyrlv1[j, k, i] > diff_pyrlv1[j + dj, k + dk, i + di]
							min_bool = min_bool and diff_pyrlv1[j, k, i] < diff_pyrlv1[j + dj, k + dk, i + di]

							if not max_bool and not min_bool:
								break
						if not max_bool and not min_bool:
							break
					if not max_bool and not min_bool:
						break

				if max_bool or min_bool:
					dx = (diff_pyrlv1[j, k + 1, i] - diff_pyrlv1[j, k - 1, i]) * 0.5 / 255
					dy = (diff_pyrlv1[j + 1, k, i] - diff_pyrlv1[j - 1, k, i]) * 0.5 / 255
					ds = (diff_pyrlv1[j, k, i + 1] - diff_pyrlv1[j, k, i - 1]) * 0.5 / 255
					dxx = (diff_pyrlv1[j, k+1, i] + diff_pyrlv1[j, k-1, i] - 2 * diff_pyrlv1[j, k, i]) * 1.0 / 255
					dyy = (diff_pyrlv1[j+1, k, i] + diff_pyrlv1[j-1, k, i] - 2 * diff_pyrlv1[j, k, i]) * 1.0 / 255
					dss = (diff_pyrlv1[j, k, i+1] + diff_pyrlv1[j, k, i-1] - 2 * diff_pyrlv1[j, k, i]) * 1.0 / 255
					dxy = (diff_pyrlv1[j+1, k+1, i] - diff_pyrlv1[j+1, k-1, i] - diff_pyrlv1[j-1, k+1, i] + diff_pyrlv1[j-1, k-1, i]) * 0.25 / 255 
					dxs = (diff_pyrlv1[j, k+1, i+1] - diff_pyrlv1[j, k-1, i+1] - diff_pyrlv1[j, k+1, i-1] + diff_pyrlv1[j, k-1, i-1]) * 0.25 / 255 
					dys = (diff_pyrlv1[j+1, k, i+1] - diff_pyrlv1[j-1, k, i+1] - diff_pyrlv1[j+1, k, i-1] + diff_pyrlv1[j-1, k, i-1]) * 0.25 / 255

					d = np.matrix([[dx], [dy], [ds]])
					H = np.matrix([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
					x = numpy.linalg.lstsq(H,d)[0]
					D_x = diff_pyrlv1[j, k, i] + 0.5 * np.dot(d.transpose(), x)

					r = 10.0
					if (((dxx + dyy) ** 2) * r) < ((r + 1) ** 2) * (dxx * dyy - (dxy ** 2)) and (np.absolute(x[0]) < 0.5) and(np.absolute(x[1]) < 0.5) and (np.absolute(x[2]) < 0.5) and (np.absolute(D_x) > 0.03) :
						extrPointLv1[j, k, i - 1] = 1


	print "Second Octave"

	for i in range(1, 4):
		for j in range(40, normal_image.shape[0] - 40):
			for k in range(40, normal_image.shape[1] - 40):
				if(np.absolute(diff_pyrlv2[j, k, i]) < threshold):
					continue

				max_bool = diff_pyrlv2[j, k, i] > 0
				min_bool = diff_pyrlv2[j, k, i] < 0

				for di in range(-1, 2):
					for dj in range(-1, 2):
						for dk in range(-1, 2):
							if di == 0 and dj == 0  and dk == 0:
								continue

							max_bool = max_bool and diff_pyrlv2[j, k, i] > diff_pyrlv2[j + dj, k + dk, i + di]
							min_bool = min_bool and diff_pyrlv2[j, k, i] < diff_pyrlv2[j + dj, k + dk, i + di]

							if not max_bool and not min_bool:
								break
						if not max_bool and not min_bool:
							break
					if not max_bool and not min_bool:
						break

				if max_bool or min_bool:
					dx = (diff_pyrlv2[j, k + 1, i] - diff_pyrlv2[j, k - 1, i]) * 0.5 / 255
					dy = (diff_pyrlv2[j + 1, k, i] - diff_pyrlv2[j - 1, k, i]) * 0.5 / 255
					ds = (diff_pyrlv2[j, k, i + 1] - diff_pyrlv2[j, k, i - 1]) * 0.5 / 255
					dxx = (diff_pyrlv2[j, k+1, i] + diff_pyrlv2[j, k-1, i] - 2 * diff_pyrlv2[j, k, i]) * 1.0 / 255        
					dyy = (diff_pyrlv2[j+1, k, i] + diff_pyrlv2[j-1, k, i] - 2 * diff_pyrlv2[j, k, i]) * 1.0 / 255          
					dss = (diff_pyrlv2[j, k, i+1] + diff_pyrlv2[j, k, i-1] - 2 * diff_pyrlv2[j, k, i]) * 1.0 / 255
					dxy = (diff_pyrlv2[j+1, k+1, i] - diff_pyrlv2[j+1, k-1, i] - diff_pyrlv2[j-1, k+1, i] + diff_pyrlv2[j-1, k-1, i]) * 0.25 / 255 
					dxs = (diff_pyrlv2[j, k+1, i+1] - diff_pyrlv2[j, k-1, i+1] - diff_pyrlv2[j, k+1, i-1] + diff_pyrlv2[j, k-1, i-1]) * 0.25 / 255 
					dys = (diff_pyrlv2[j+1, k, i+1] - diff_pyrlv2[j-1, k, i+1] - diff_pyrlv2[j+1, k, i-1] + diff_pyrlv2[j-1, k, i-1]) * 0.25 / 255  

					d = np.matrix([[dx], [dy], [ds]])
					H = np.matrix([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
					x = np.linalg.lstsq(H,d)[0]
					D_x = diff_pyrlv2[j, k, i] + 0.5 * np.dot(d.transpose(), x)

					r = 10.0
					if (((dxx + dyy) ** 2) * r) < ((r + 1) ** 2) * (dxx * dyy - (dxy ** 2)) and (np.absolute(x[0]) < 0.5) and(np.absolute(x[1]) < 0.5) and (np.absolute(x[2]) < 0.5) and (np.absolute(D_x) > 0.03) :
						extrPointLv2[j, k, i - 1] = 1

	print "Third Octave"

	for i in range(1, 4):
		for j in range(20, halved_image.shape[0] - 20):
			for k in range(20, halved_image.shape[1] - 20):
				if(np.absolute(diff_pyrlv3[j, k, i]) < threshold):
					continue

				max_bool = diff_pyrlv3[j, k, i] > 0
				min_bool = diff_pyrlv3[j, k, i] < 0

				for di in range(-1, 2):
					for dj in range(-1, 2):
						for dk in range(-1, 2):
							if di == 0 and dj == 0  and dk == 0:
								continue

							max_bool = max_bool and diff_pyrlv3[j, k, i] > diff_pyrlv3[j + dj, k + dk, i + di]
							min_bool = min_bool and diff_pyrlv3[j, k, i] < diff_pyrlv3[j + dj, k + dk, i + di]

							if not max_bool and not min_bool:
								break
						if not max_bool and not min_bool:
							break
					if not max_bool and not min_bool:
						break

				if max_bool or min_bool:
					dx = (diff_pyrlv3[j, k + 1, i] - diff_pyrlv3[j, k - 1, i]) * 0.5 / 255
					dy = (diff_pyrlv3[j + 1, k, i] - diff_pyrlv3[j - 1, k, i]) * 0.5 / 255
					ds = (diff_pyrlv3[j, k, i + 1] - diff_pyrlv3[j, k, i - 1]) * 0.5 / 255
					dxx = (diff_pyrlv3[j, k+1, i] + diff_pyrlv3[j, k-1, i] - 2 * diff_pyrlv3[j, k, i]) * 1.0 / 255        
					dyy = (diff_pyrlv3[j+1, k, i] + diff_pyrlv3[j-1, k, i] - 2 * diff_pyrlv3[j, k, i]) * 1.0 / 255          
					dss = (diff_pyrlv3[j, k, i+1] + diff_pyrlv3[j, k, i-1] - 2 * diff_pyrlv3[j, k, i]) * 1.0 / 255
					dxy = (diff_pyrlv3[j+1, k+1, i] - diff_pyrlv3[j+1, k-1, i] - diff_pyrlv3[j-1, k+1, i] + diff_pyrlv3[j-1, k-1, i]) * 0.25 / 255 
					dxs = (diff_pyrlv3[j, k+1, i+1] - diff_pyrlv3[j, k-1, i+1] - diff_pyrlv3[j, k+1, i-1] + diff_pyrlv3[j, k-1, i-1]) * 0.25 / 255 
					dys = (diff_pyrlv3[j+1, k, i+1] - diff_pyrlv3[j-1, k, i+1] - diff_pyrlv3[j+1, k, i-1] + diff_pyrlv3[j-1, k, i-1]) * 0.25 / 255  

					d = np.matrix([[dx], [dy], [ds]])
					H = np.matrix([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
					x = np.linalg.lstsq(H,d)[0]
					D_x = diff_pyrlv3[j, k, i] + 0.5 * np.dot(d.transpose(), x)

					r = 10.0
					if (((dxx + dyy) ** 2) * r) < ((r + 1) ** 2) * (dxx * dyy - (dxy ** 2)) and (np.absolute(x[0]) < 0.5) and(np.absolute(x[1]) < 0.5) and (np.absolute(x[2]) < 0.5) and (np.absolute(D_x) > 0.03) :
						extrPointLv3[j, k, i - 1] = 1

	print "Fourth Octave"

	for i in range(1, 4):
		for j in range(10, quarted_image.shape[0] - 10):
			for k in range(10, quarted_image.shape[1] - 10):
				if(np.absolute(diff_pyrlv4[j, k, i]) < threshold):
					continue

				max_bool = diff_pyrlv4[j, k, i] > 0
				min_bool = diff_pyrlv4[j, k, i] < 0

				for di in range(-1, 2):
					for dj in range(-1, 2):
						for dk in range(-1, 2):
							if di == 0 and dj == 0  and dk == 0:
								continue

							max_bool = max_bool and diff_pyrlv4[j, k, i] > diff_pyrlv4[j + dj, k + dk, i + di]
							min_bool = min_bool and diff_pyrlv4[j, k, i] < diff_pyrlv4[j + dj, k + dk, i + di]

							if not max_bool and not min_bool:
								break
						if not max_bool and not min_bool:
							break
					if not max_bool and not min_bool:
						break

				if max_bool or min_bool:
					dx = (diff_pyrlv4[j, k + 1, i] - diff_pyrlv4[j, k - 1, i]) * 0.5 / 255
					dy = (diff_pyrlv4[j + 1, k, i] - diff_pyrlv4[j - 1, k, i]) * 0.5 / 255
					ds = (diff_pyrlv4[j, k, i + 1] - diff_pyrlv4[j, k, i - 1]) * 0.5 / 255
					dxx = (diff_pyrlv4[j, k+1, i] + diff_pyrlv4[j, k-1, i] - 2 * diff_pyrlv4[j, k, i]) * 1.0 / 255        
					dyy = (diff_pyrlv4[j+1, k, i] + diff_pyrlv4[j-1, k, i] - 2 * diff_pyrlv4[j, k, i]) * 1.0 / 255          
					dss = (diff_pyrlv4[j, k, i+1] + diff_pyrlv4[j, k, i-1] - 2 * diff_pyrlv4[j, k, i]) * 1.0 / 255
					dxy = (diff_pyrlv4[j+1, k+1, i] - diff_pyrlv4[j+1, k-1, i] - diff_pyrlv4[j-1, k+1, i] + diff_pyrlv4[j-1, k-1, i]) * 0.25 / 255 
					dxs = (diff_pyrlv4[j, k+1, i+1] - diff_pyrlv4[j, k-1, i+1] - diff_pyrlv4[j, k+1, i-1] + diff_pyrlv4[j, k-1, i-1]) * 0.25 / 255 
					dys = (diff_pyrlv4[j+1, k, i+1] - diff_pyrlv4[j-1, k, i+1] - diff_pyrlv4[j+1, k, i-1] + diff_pyrlv4[j-1, k, i-1]) * 0.25 / 255  

					d = np.matrix([[dx], [dy], [ds]])
					H = np.matrix([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
					x = np.linalg.lstsq(H,d)[0]
					D_x = diff_pyrlv4[j, k, i] + 0.5 * np.dot(d.transpose(), x)

					r = 10.0
					if (((dxx + dyy) ** 2) * r) < ((r + 1) ** 2) * (dxx * dyy - (dxy ** 2)) and (np.absolute(x[0]) < 0.5) and(np.absolute(x[1]) < 0.5) and (np.absolute(x[2]) < 0.5) and (np.absolute(D_x) > 0.03) :
						extrPointLv4[j, k, i - 1] = 1


	print "Number of extreme points in pyramid level1 %d" % np.sum(extrPointLv1)
	print "Number of extreme points in pyramid level2 %d" % np.sum(extrPointLv2)
	print "Number of extreme points in pyramid level3 %d" % np.sum(extrPointLv3)
	print "Number of extreme points in pyramid level4 %d" % np.sum(extrPointLv4)


	mag_pyrlv1 = np.zeros((doubled_image.shape[0], doubled_image.shape[1], 3))
	mag_pyrlv2 = np.zeros((normal_image.shape[0], normal_image.shape[1], 3))
	mag_pyrlv3 = np.zeros((halved_image.shape[0], halved_image.shape[1], 3))
	mag_pyrlv4 = np.zeros((quarted_image.shape[0], quarted_image.shape[1], 3))

	ori_pyrlv1 = np.zeros((doubled_image.shape[0], doubled_image.shape[1], 3))
	ori_pyrlv2 = np.zeros((normal_image.shape[0], normal_image.shape[1], 3))
	ori_pyrlv3 = np.zeros((halved_image.shape[0], halved_image.shape[1], 3))
	ori_pyrlv4 = np.zeros((quarted_image.shape[0], quarted_image.shape[1], 3))

	for i in range(3):
		for j in range(1, doubled_image.shape[0] - 1):
			for k in range(1, doubled_image.shape[1] - 1):
				mag_pyrlv1[j, k, i] = (((doubled_image[j + 1, k] - doubled_image[j - 1, k]) ** 2) + ((doubled_image[j, k + 1] - doubled_image[j, k - 1]) ** 2)) ** 0.5
				ori_pyrlv1[j, k, i] = (36 / 2 * np.pi) * (np.pi + np.arctan2((doubled_image[j, k + 1] - doubled_image[j, k - 1]), (doubled_image[j + 1, k] - doubled_image[j - 1, k])))

	for i in range(3):
		for j in range(1, normal_image.shape[0] - 1):
			for k in range(1, normal_image.shape[1] - 1):
				mag_pyrlv2[j, k, i] = (((normal_image[j + 1, k] - normal_image[j - 1, k]) ** 2) + ((normal_image[j, k + 1] - normal_image[j, k - 1]) ** 2)) ** 0.5
				ori_pyrlv2[j, k, i] = (36 / 2 * np.pi) * (np.pi + np.arctan2((normal_image[j, k + 1] - normal_image[j, k - 1]), (normal_image[j + 1, k] - normal_image[j - 1, k])))

	for i in range(3):
		for j in range(1, halved_image.shape[0] - 1):
			for k in range(1, halved_image.shape[1] - 1):
				mag_pyrlv3[j, k, i] = (((halved_image[j + 1, k] - halved_image[j - 1, k]) ** 2) + ((halved_image[j, k + 1] - halved_image[j, k - 1]) ** 2)) ** 0.5
				ori_pyrlv3[j, k, i] = (36 / 2 * np.pi) * (np.pi + np.arctan2((halved_image[j, k + 1] - halved_image[j, k - 1]), (halved_image[j + 1, k] - halved_image[j - 1, k])))

	for i in range(3):
		for j in range(1, quarted_image.shape[0] - 1):
			for k in range(1, quarted_image.shape[1] - 1):
				mag_pyrlv4[j, k, i] = (((quarted_image[j + 1, k] - quarted_image[j - 1, k]) ** 2) + ((quarted_image[j, k + 1] - quarted_image[j, k - 1]) ** 2)) ** 0.5
				ori_pyrlv4[j, k, i] = (36 / 2 * np.pi) * (np.pi + np.arctan2((quarted_image[j, k + 1] - quarted_image[j, k - 1]), (quarted_image[j + 1, k] - quarted_image[j - 1, k])))

	sumPoints = np.sum(extrPointLv1) + np.sum(extrPointLv2) + np.sum(extrPointLv3) + np.sum(extrPointLv4)

	keyPoints = np.zeros((sumPoints, 4))


	print "Calculating keypoints orientation"

	count = 0

	for i in range(3):
		for j in range(80, doubled_image.shape[0] - 80):
			for k in range(80, doubled_image.shape[1] - 80):
				if extrPointLv1[j, k, i] == 1:
					gaussian_window = multivariate_normal(mean = [j, k], cov = ((1.5 * paramtotal[i]) ** 2))
					radius = 2 * np.floor(2 * 1.5 * paramtotal[i])
					
					orient_hist = np.zeros([36, 1])

					for x in range(int(-1 * radius), int(radius + 1)):
						y_range = int((radius ** 2 - (np.absolute(x) ** 2)) ** 0.5)
						for y in range(-1 * y_range, y_range + 1):
							
							if j + x < 0 or j + x > doubled_image.shape[0] - 1 or k + y < 0 or k + y > doubled_image.shape[1] - 1:
								continue

							weight = mag_pyrlv1[j + x, k + y, i] * gaussian_window.pdf([j + x, k + y])
							bin_indx = np.clip(np.floor(ori_pyrlv1[j + x, k + y, i]), 0, 35)
							orient_hist[np.floor(bin_indx)] += weight

					max_val = np.amax(orient_hist)
					max_index = np.argmax(orient_hist)

					keyPoints[count, :] = np.array([int(j * 0.5), int(k * 0.5), paramtotal[i], max_index])
					count += 1

					orient_hist[max_index] = 0
					new_max = np.amax(orient_hist)
					while new_max > 0.8 * max_val:
						new_index = np.argmax(orient_hist)
						np.append(keyPoints, np.array([[int(j * 0.5), int(k * 0.5), paramtotal[i], new_index]]), axis = 0)
						orient_hist[new_index] = 0
						new_max = np.amax(orient_hist)


	for i in range(3):
		for j in range(40, normal_image.shape[0] - 40):
			for k in range(40, normal_image.shape[1] - 40):
				if extrPointLv2[j, k, i] == 1:
					gaussian_window = multivariate_normal(mean = [j, k], cov = ((1.5 * paramtotal[i + 3]) ** 2))
					radius = 2 * np.floor(2 * 1.5 * paramtotal[i + 3])
					
					orient_hist = np.zeros([36, 1])

					for x in range(int(-1 * radius), int(radius + 1)):
						y_range = int((radius ** 2 - (np.absolute(x) ** 2)) ** 0.5)
						for y in range(-1 * y_range, y_range + 1):
							
							if j + x < 0 or j + x > normal_image.shape[0] - 1 or k + y < 0 or k + y > normal_image.shape[1] - 1:
								continue

							weight = mag_pyrlv2[j + x, k + y, i] * gaussian_window.pdf([j + x, k + y])
							bin_indx = np.clip(np.floor(ori_pyrlv2[j + x, k + y, i]), 0, 35)
							orient_hist[np.floor(bin_indx)] += weight

					max_val = np.amax(orient_hist)
					max_index = np.argmax(orient_hist)

					keyPoints[count, :] = np.array([j, k, paramtotal[i + 3], max_index])
					count += 1

					orient_hist[max_index] = 0
					new_max = np.amax(orient_hist)
					while new_max > 0.8 * max_val:
						new_index = np.argmax(orient_hist)
						np.append(keyPoints, np.array([[j, k, paramtotal[i + 3], new_index]]), axis = 0)
						orient_hist[new_index] = 0
						new_max = np.amax(orient_hist)


	for i in range(3):
		for j in range(20, halved_image.shape[0] - 20):
			for k in range(20, halved_image.shape[1] - 20):
				if extrPointLv3[j, k, i] == 1:
					gaussian_window = multivariate_normal(mean = [j, k], cov = ((1.5 * paramtotal[i + 6]) ** 2))
					radius = 2 * np.floor(2 * 1.5 * paramtotal[i + 6])
					
					orient_hist = np.zeros([36, 1])

					for x in range(int(-1 * radius), int(radius + 1)):
						y_range = int((radius ** 2 - (np.absolute(x) ** 2)) ** 0.5)
						for y in range(-1 * y_range, y_range + 1):
							if j + x < 0 or j + x > halved_image.shape[0] - 1 or k + y < 0 or k + y > halved_image.shape[1] - 1:
								continue

							weight = mag_pyrlv3[j + x, k + y, i] * gaussian_window.pdf([j + x, k + y])
							bin_indx = np.clip(np.floor(ori_pyrlv3[j + x, k + y, i]), 0, 35)
							orient_hist[np.floor(bin_indx)] += weight

					max_val = np.amax(orient_hist)
					max_index = np.argmax(orient_hist)

					keyPoints[count, :] = np.array([j * 2, k * 2, paramtotal[i + 6], max_index])
					count += 1

					orient_hist[max_index] = 0
					new_max = np.amax(orient_hist)
					while new_max > 0.8 * max_val:
						new_index = np.argmax(orient_hist)
						np.append(keyPoints, np.array([[j * 2, k * 2, paramtotal[i + 6], new_index]]), axis = 0)
						orient_hist[new_index] = 0
						new_max = np.amax(orient_hist)


	for i in range(3):
		for j in range(10, quarted_image.shape[0] - 10):
			for k in range(10, quarted_image.shape[1] - 10):
				if extrPointLv4[j, k, i] == 1:
					gaussian_window = multivariate_normal(mean = [j, k], cov = ((1.5 * paramtotal[i + 9]) ** 2))
					radius = 2 * np.floor(2 * 1.5 * paramtotal[i + 9])
					
					orient_hist = np.zeros([36, 1])

					for x in range(int(-1 * radius), int(radius + 1)):
						y_range = int((radius ** 2 - (np.absolute(x) ** 2)) ** 0.5)
						for y in range(-1 * y_range, y_range + 1):
							
							if j + x < 0 or j + x > quarted_image.shape[0] - 1 or k + y < 0 or k + y > quarted_image.shape[1] - 1:
								continue

							weight = mag_pyrlv4[j + x, k + y, i] * gaussian_window.pdf([j + x, k + y])
							bin_indx = np.clip(np.floor(ori_pyrlv4[j + x, k + y, i]), 0, 35)
							orient_hist[np.floor(bin_indx)] += weight

					max_val = np.amax(orient_hist)
					max_index = np.argmax(orient_hist)

					keyPoints[count, :] = np.array([int(j * 4), int(k * 4), paramtotal[i + 9], max_index])
					count += 1

					orient_hist[max_index] = 0
					new_max = np.amax(orient_hist)
					while new_max > 0.8 * max_val:
						new_index = np.argmax(orient_hist)
						np.append(keyPoints, np.array([[int(j * 4), int(k * 4), paramtotal[i + 9], new_index]]), axis = 0)
						orient_hist[new_index] = 0
						new_max = np.amax(orient_hist)


	print "Calculating desciptor"

	magpyr = np.zeros((normal_image.shape[0], normal_image.shape[1], 12))
	oripyr = np.zeros((normal_image.shape[0], normal_image.shape[1], 12))

	for i in range(3):
		magmax = np.amax(mag_pyrlv1[:, :, i])
		magpyr[:, :, i] = misc.imresize(mag_pyrlv1[:, :, i], (normal_image.shape[0], normal_image.shape[1]), "bilinear").astype(float)
		magpyr[:, :, i] = (magmax / np.amax(magpyr[:, :, i])) * (magpyr[:, :, i])
		oripyr[:, :, i] = misc.imresize(ori_pyrlv1[:, :, i], (normal_image.shape[0], normal_image.shape[1]), "bilinear").astype(int)
		oripyr[:, :, i] = (36.0 / np.amax(oripyr[:, :, i])) * oripyr[:, :, i].astype(int)

	for i in range(3):
		magpyr[:, :, i + 3] = mag_pyrlv2[:, :, i].astype(float)
		oripyr[:, :, i + 3] = ori_pyrlv2[:, :, i].astype(int)

	for i in range(3):
		magpyr[:, :, i + 6] = misc.imresize(mag_pyrlv3[:, :, i], (normal_image.shape[0], normal_image.shape[1]), "bilinear").astype(int)
		oripyr[:, :, i + 6] = misc.imresize(ori_pyrlv3[:, :, i], (normal_image.shape[0], normal_image.shape[1]), "bilinear").astype(int)

	for i in range(3):
		magpyr[:, :, i + 9] = misc.imresize(mag_pyrlv4[:, :, i], (normal_image.shape[0], normal_image.shape[1]), "bilinear").astype(int)
		oripyr[:, :, i + 9] = misc.imresize(ori_pyrlv4[:, :, i], (normal_image.shape[0], normal_image.shape[1]), "bilinear").astype(int)

	descriptors = np.zeros([keyPoints.shape[0], 128])

	for i in range(0, keyPoints.shape[0]):
		for x in range(-8, 8):
			for y in range(-8, 8):
				theta = 10.0 * keyPoints[i, 3] * np.pi / 180.0
				xrot = np.round((np.cos(theta) * x) - (np.sin(theta) * y))
				yrot = np.round((np.sin(theta) * x) + (np.cos(theta) * y))

				scale_index = np.argwhere(paramtotal == keyPoints[i, 2])[0][0]

				x0 = keyPoints[i, 0]
				y0 = keyPoints[i, 1]

				gaussian_window = multivariate_normal(mean = [x0, y0], cov = 8)
				weight = magpyr[x0 + xrot, y0 + yrot, scale_index] * gaussian_window.pdf([x0 + xrot, y0 + yrot])
				angle = oripyr[x0 + xrot, y0 + yrot, scale_index] - keyPoints[i, 3]
				if angle < 0:
					angle += 36

				bin_index = np.clip(np.floor(8.0/ 36.0 * angle), 0, 7).astype(int)
				descriptors[i, 32 * int((x + 8) / 4) + 8 * int((y + 8) / 4) + bin_index] += weight

		descriptors[i, :] = descriptors[i, :] / norm(descriptors[i, :])
		descriptors[i, :] = np.clip(descriptors[i, :], 0, 0.2)
		descriptors[i, :] = descriptors[i, :] / norm(descriptors[i, :])

	return [keyPoints, descriptors]

def main():
	img1_name = "/home/yihong/Desktop/1.png"
	img2_name = "/home/yihong/Desktop/input2.tif"

	detect_features(img1_name, 5)


if __name__ == "__main__":
	main()
