import cv2
import os
import math
from scipy.spatial import distance as dist
from sklearn.metrics import mean_absolute_error

# import the necessary packages

from collections import OrderedDict
import numpy as np

class CentroidTracker():
	def __init__(self, maxDisappeared=5):
		# initialize the next unique object ID along with two ordered
		# dictionaries used to keep track of mapping a given object
		# ID to its centroid and number of consecutive frames it has
		# been marked as "disappeared", respectively
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()

		# store the number of maximum consecutive frames a given
		# object is allowed to be marked as "disappeared" until we
		# need to deregister the object from tracking
		self.maxDisappeared = maxDisappeared

	def register(self, centroid):
		# when registering an object we use the next available object
		# ID to store the centroid
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1

	def deregister(self, objectID):
		# to deregister an object ID we delete the object ID from
		# both of our respective dictionaries
		del self.objects[objectID]
		del self.disappeared[objectID]

	def update(self, rects):
		# check to see if the list of input bounding box rectangles
		# is empty
		if len(rects) == 0:
			# loop over any existing tracked objects and mark them
			# as disappeared
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1

				# if we have reached a maximum number of consecutive
				# frames where a given object has been marked as
				# missing, deregister it
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)

			# return early as there are no centroids or tracking info
			# to update
			return self.objects

		# initialize an array of input centroids for the current frame
		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		# loop over the bounding box rectangles
		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			# use the bounding box coordinates to derive the centroid
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)

		# if we are currently not tracking any objects take the input
		# centroids and register each of them
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i])

		# otherwise, are are currently tracking objects so we need to
		# try to match the input centroids to existing object
		# centroids
		else:
			# grab the set of object IDs and corresponding centroids
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())

			# compute the distance between each pair of object
			# centroids and input centroids, respectively -- our
			# goal will be to match an input centroid to an existing
			# object centroid
			D = dist.cdist(np.array(objectCentroids), inputCentroids)

			# in order to perform this matching we must (1) find the
			# smallest value in each row and then (2) sort the row
			# indexes based on their minimum values so that the row
			# with the smallest value as at the *front* of the index
			# list
			rows = D.min(axis=1).argsort()

			# next, we perform a similar process on the columns by
			# finding the smallest value in each column and then
			# sorting using the previously computed row index list
			cols = D.argmin(axis=1)[rows]

			# in order to determine if we need to update, register,
			# or deregister an object we need to keep track of which
			# of the rows and column indexes we have already examined
			usedRows = set()
			usedCols = set()

			# loop over the combination of the (row, column) index
			# tuples
			for (row, col) in zip(rows, cols):
				# if we have already examined either the row or
				# column value before, ignore it
				# val
				if row in usedRows or col in usedCols:
					continue

				# otherwise, grab the object ID for the current row,
				# set its new centroid, and reset the disappeared
				# counter
				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0

				# indicate that we have examined each of the row and
				# column indexes, respectively
				usedRows.add(row)
				usedCols.add(col)

			# compute both the row and column index we have NOT yet
			# examined
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			# in the event that the number of object centroids is
			# equal or greater than the number of input centroids
			# we need to check and see if some of these objects have
			# potentially disappeared
			if D.shape[0] >= D.shape[1]:
				# loop over the unused row indexes
				for row in unusedRows:
					# grab the object ID for the corresponding row
					# index and increment the disappeared counter
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1

					# check to see if the number of consecutive
					# frames the object has been marked "disappeared"
					# for warrants deregistering the object
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)

			# otherwise, if the number of input centroids is greater
			# than the number of existing object centroids we need to
			# register each new input centroid as a trackable object
			else:
				for col in unusedCols:
					self.register(inputCentroids[col])

		# return the set of trackable objects
		return self.objects
def ucitavanje_putanja():
    directory = r'C:\Users\Nikola\Desktop\VIDEO'
    lst = []
    for entry in os.scandir(directory):
        if entry.path.endswith(".mp4") and entry.is_file():
            lst.append(entry.path)

    return lst


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def nadji_pravouganoik(tacke):
    t1 = (0, 0)
    t2 = (0, 0)
    max_duz = 0
    for i in range(len(tacke) - 1):
        m1 = tacke[i]
        for j in range(i + 1, len(tacke)):
            m2 = tacke[j]
            duzina = math.sqrt(pow(m2[0] - m1[0], 2) + pow(m2[1] - m1[1], 2))

            if duzina > max_duz:
                max_duz = duzina
                t1 = m1
                t2 = m2

    tacke.remove(t1)
    tacke.remove(t2)

    return tacke


def presecne_tacke(lines, path):
    p_tacke = []

    cap = cv2.VideoCapture(path)
    ret_val, frame = cap.read()
    for i in range(len(lines)):
        ax = lines[i][0][0]

        ay = lines[i][0][1]

        bx = lines[i][0][2]
        by = lines[i][0][3]
        #cv2.circle(frame, (ax, ay), 10, (255, 0, 0), -1)
        #cv2.circle(frame, (bx, by), 10, (255, 0, 0), -1)

    # cv2.circle(frame, (461, 93), 10, (0, 0, 255), -1)
    cv2.circle(frame, (171, 113), 10, (0, 0, 255), -1)
    # cv2.circle(frame, (148, 463), 10, (0, 0, 255), -1)
    cv2.circle(frame, (516, 437), 10, (0, 0, 255), -1)
    #cv2.imshow("TACKE", frame)
    #cv2.waitKey()
    cap.release()

    for i in range(0, len(lines) - 1):
        ax = lines[i][0][0]
        ay = lines[i][0][1]
        bx = lines[i][0][2]
        by = lines[i][0][3]

        if ay < 70 and by < 70:
            continue

        for j in range(i + 1, len(lines)):
            cx = lines[j][0][0]
            cy = lines[j][0][1]
            dx = lines[j][0][2]
            dy = lines[j][0][3]

            if cy < 70 and dy < 70:
                continue

            A = (ax, ay)
            B = (bx, by)
            C = (cx, cy)
            D = (dx, dy)

            presecna_tacka = line_intersection((A, B), (C, D))
            r1 = round(presecna_tacka[0], -1)
            r2 = round(presecna_tacka[1], -1)
            presecna_tacka = (r1, r2)

            if (not presecna_tacka[0] < 0 and not presecna_tacka[0] > 640 and not presecna_tacka[1] < 0
                    and not presecna_tacka[1] > 480):
                p_tacke.append(presecna_tacka)

    p_tacke_final = []

    for tacka in p_tacke:
        if tacka not in p_tacke_final:
            p_tacke_final.append(tacka)

    return p_tacke_final


def pronadji_liniju(path):
    cap = cv2.VideoCapture(path)
    ret_val, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 110, apertureSize=3)
    lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 180, threshold=165, lines=np.array([]), minLineLength=300,
                            maxLineGap=20)
    #print("BROJ LINIJA " + str(len(lines)))

    for i in range(len(lines)):
        cv2.line(frame, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 255, 0), 3);

    #cv2.imshow("LINIJE", frame)
    #cv2.waitKey(0)
    cap.release()
    return lines

def minResenje(y):
    y_line = (110 +90) /2
    if abs(y - y_line) < 7.6:
        # print("Pedestrian detected")
        return True
    else:
        return False


def detektuj_ljude(stari_frame, novi_frame,gornja_leva,donja_desna,iskorisceni_kljucevi,counter):


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    diff = cv2.absdiff(stari_frame, novi_frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    frame = cv2.adaptiveThreshold(blur, 155, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 5)

    dilated = cv2.dilate(frame, kernel, iterations=3)
    eroded = cv2.erode(dilated, kernel, iterations=3)

    #cv2.imshow('th', dilated)
    #cv2.waitKey(0)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    lista = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        t1 = (x, y)
        t2 = (x + w, y + h)
        if w > 15 and w < 50 and h > 25 and h < 50:
            lista.append((x, y, x + w, y + h))


    #cv2.imshow('new', stari_frame)
    #cv2.waitKey()
    objects = ct.update(lista)


    for key in objects.keys():
        centar_coveka = objects.get(key)
        if centar_coveka[0]>gornja_leva[0] and centar_coveka[1]>gornja_leva[1] and centar_coveka[0]<donja_desna[0] and centar_coveka[1]<donja_desna[1]:
            if key not in iskorisceni_kljucevi:
                counter +=1
                iskorisceni_kljucevi.append(key)








    return counter


def video_zapis(ct,path, tacke_pravougaonika):

    f = open("C:\\Users\\Nikola\\Desktop\\VIDEO\\res.txt", "r")
    naziv_videa = path[30:]
    print(naziv_videa)
    string = f.readline()
    nazivi = []
    brojevi = []
    for x in f:
        x = x.strip('\n')
        razdvojeno = x.split(",")
        nazivi.append(razdvojeno[0])
        brojevi.append(razdvojeno[1])



    cap = cv2.VideoCapture(path)
    counter = 0
    iskorisceni_kljucevi = []
    gornja_leva = tacke_pravougaonika[0]
    donja_desna = tacke_pravougaonika[1]
    r1 = round(gornja_leva[0])
    r2 = round(gornja_leva[1])
    gornja_leva = (r1, r2)

    r1 = round(donja_desna[0])
    r2 = round(donja_desna[1])
    donja_desna = (r1, r2)
    tacke_r = []
    tacke_r.append(gornja_leva)
    tacke_r.append(donja_desna)




    ret_val, frame = cap.read()


    while True:
        stari_frame = frame.copy()
        ret_val, frame = cap.read()
        if not ret_val:
            break
        novi_frame = frame.copy()

        counter = detektuj_ljude(stari_frame,novi_frame,gornja_leva,donja_desna,iskorisceni_kljucevi,counter)


    indeks = 0;


    
    for i in range(len(nazivi)):
        if nazivi[i]==naziv_videa:
            indeks=i
            break


    prava_vrednost = 0
    broj = brojevi.__getitem__(i)


    print("Broj ljudi iznosi: "+str(counter))
    broj = int(broj)
    mae_original = [broj]
    mae_calc= [counter]

    mae = mean_absolute_error(mae_original,mae_calc)
    print("AE za video "+naziv_videa+" iznosi: "+str(mae))

    return counter


if __name__ == '__main__':
    suma_const = 196
    lista_putanja = ucitavanje_putanja()
    path = lista_putanja[0]
    lines = pronadji_liniju(path)
    tacke = presecne_tacke(lines, path)
    tacke_pravougaonika = nadji_pravouganoik(tacke)
    lista_final = [4,23, 24, 17, 23, 17, 27, 29, 22, 10]
    lista_mae = []

    for x in lista_putanja:
        ct = CentroidTracker()
        (H, W) = (640, 480)
        counter = video_zapis(ct,x, tacke_pravougaonika)
        print("===================================")
        lista_mae.append(counter)



    mae = mean_absolute_error(lista_final,lista_mae)
    print("Ukupan MAE iznosi " + str(mae))