from matplotlib import pyplot as plt
import cv2
import numpy as np

train_num = 7202
val_num = 522

def BW(total, set):
	for i in range(total):
		path = '/Users/huiwenyou/Desktop/hack/pics/cell_images/' + set + '/' + str(i+1) + '.jpg'
		pic = plt.imread(path)/255 # dividing by 255 to bring the pixel values between 0 and 1
		# plt.imshow(pic)

		pic_n = pic.reshape(pic.shape[0]*pic.shape[1], pic.shape[2])
		# print(pic_n.shape)

		from sklearn.cluster import KMeans
		kmeans = KMeans(n_clusters=2, random_state=0).fit(pic_n)
		# print(kmeans.labels_)
		pic2show = kmeans.cluster_centers_[kmeans.labels_]
		# np.histogram(pic2show)
		# plt.hist(pic2show[:, 0], bins='auto')
		# plt.show()
		flattened = pic2show.flatten()
		mean = (max(flattened) + min(flattened)) / 2
		for r, each_row in enumerate(pic2show):
			for c, col in enumerate(each_row):
				if col > mean:
					pic2show[r, c] = 1
				else:
					pic2show[r, c] = 0

		cluster_pic = pic2show.reshape(pic.shape[0], pic.shape[1], pic.shape[2])
		path1 = '/Users/huiwenyou/Desktop/hack/pics/cell_images/' + set + '/BW/clean' + str(i + 1) + '.jpg'
		cv2.imwrite(path1, cluster_pic * 255)
	# print("done")

def overlap(rect, rest):
	for rect_each in rest:
		_x, _y, _w, _h = rect_each
		x, y, w, h = rect
		if x + w <= _x + _w and x > _x and y + h <= _y + _h and y >= _y:
			# inside
			print("{} is inside {}".format(rect, rect_each))
			return True
	return False

def segment(total_num):
	# segmentation version 2
	for i in range(total_num):
		path = '/Users/huiwenyou/Desktop/hack/pics/cell_images/' + set_name + '/BW/clean' + str(i+1) + '.jpg'
		src = cv2.imread(path, 1) # read input image 3 color
		height, width, channels = src.shape
		area = height * width

		gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) # convert to grayscale
		blur = cv2.blur(gray, (3, 3)) # blur the image
		ret, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

		contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		# create hull array for convex hull points
		hull = []
		hull_vertices = []

		rect = []
		rect_vertices = []

		drawing_rect = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
		# calculate points for each contour
		for ci in range(len(contours)):
			# creating convex hull object for each contour
			hull_vertices.append(cv2.convexHull(contours[ci], clockwise=True))
			hull.append(cv2.convexHull(contours[ci], False))
			x, y, w, h = cv2.boundingRect(contours[ci])
			rect.append([x, y, w, h])

		print(area)
		filtered_rect = []
		for rect_info in rect:
			x, y, w, h = rect_info
			if w * h > 0.85 * area:
				continue
			else:
				filtered_rect.append(rect_info)

		filtered_rect_2 = []
		to_remove = []
		for rect_info in filtered_rect:
			# points_4 = [(x, y), (x, y + w), (x + h, y + w), (x + h, y)]
			rest = [can for can in filtered_rect if can != rect_info]
			is_inside = overlap(rect_info, rest)
			if is_inside:
				to_remove.append(rect_info)

		# print(filtered_rect)
		filtered_rect_2 = [can for can in filtered_rect if can not in to_remove]
		# print(filtered_rect_2)

		if plot:
			for each in filtered_rect_2:
				x, y, w, h = each
				cv2.rectangle(drawing_rect, (x, y), (x + w, y + h), (255, 0, 0), 1)

		# save splits - sort by column
		import operator, os
		filtered_rect_2.sort(key=operator.itemgetter(1))

		print(filtered_rect_2)
		for idx, sorted_each in enumerate(filtered_rect_2):
			x, y, w, h = sorted_each
			crop = src[y: y+h, x: x+w]
			out_dir = '/Users/huiwenyou/Desktop/hack/pics/cell_images/' + set_name + '/Split/f' + str(i+1) + '/'
			if not os.path.exists(out_dir):
				os.mkdir(out_dir)
			out_path = out_dir + str(i) + '_' + str(idx+1) + '.jpg'
			cv2.imwrite(out_path, crop)
			# crop.save(out_path, 'jpg')

		if plot:
			cv2.imshow("rect", drawing_rect)
			# create an empty black image
			drawing_hull = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
			# draw contours and hull points
			for i in range(len(contours)):
				color_contours = (0, 255, 0)  # green - color for contours
				color = (255, 0, 0)  # blue - color for convex hull
				# draw ith contour
				cv2.drawContours(drawing_hull, contours, i, color_contours, 1, 8, hierarchy)
				# draw ith convex hull object
				cv2.drawContours(drawing_hull, hull, i, color, 1, 8)
			cv2.imshow("hull", drawing_hull)
			cv2.waitKey()
			cv2.destroyAllWindows()



plot = True
total_num = train_num #val_num
set_name = 'training_set' #'validation_set'

# for BW
BW(total_num, set_name)
