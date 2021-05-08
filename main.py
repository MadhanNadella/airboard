import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75, min_tracking_confidence=0.5)

class Marker:
    def __init__(self):
        #index, middle, ring, pnky fingers for writing
        self.i = [[]]
        self.m = [[]]
        self.r = [[]]
        self.p = [[]]

        #few parameters for pose detection
        self.ia = 0
        self.ma = 0
        self.ra = 0
        self.pa = 0

        #if hand is there in the image
        self.ishand = 0
        #Flipping the video for a backward board
        #=1 if the board is front(or in air), =0 if the board is backward
        self.isflip = 1

    #Function to get the hand lamdmarks
    def hands_lmf(self,imag):
        if self.isflip:
            imag = cv2.flip(imag, 1)
        #Conversion to RGB colour for better image processing
        imgRGB = cv2.cvtColor(imag, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        self.ishand = 1 if results.multi_hand_landmarks else 0
        return results.multi_hand_landmarks

    #converting the landmarks to pixels in the image
    def lmf2lm(self,lmf, dim):
        lm = []
        for i in range(21):
            lm.append((int(lmf.landmark[i].x*dim[1]),int(lmf.landmark[i].y*dim[0])))
        return lm

    #Gesture controls
    def is_straight(self,lm):
        #Determining the pose of fingers(other than thumb)
        jointsi_1 = [ 6, 8]
        jointsi_2 = [ 7, 8]
        jointsm_1 = [10,12]
        jointsm_2 = [11,12]
        jointsr_1 = [14,16]
        jointsr_2 = [15,16]
        jointsp_1 = [18,20]
        jointsp_2 = [19,20]

        #pose of thumb
        jt_1 = [5,17,4]
        jt_2 = [1,17]
        jt1 = 1 if ((lm[jt_1[0]][0]-lm[jt_1[2]][0])*(lm[jt_1[1]][0]-lm[jt_1[2]][0]))>0 else 0
        jt2 = 1 if lm[jt_2[0]][1]<lm[jt_2[1]][1] else 0
        #variable to enable erasing
        self.iferase = jt1*jt2

        self.ia = (1 if lm[jointsi_1[0]][1]>lm[jointsi_1[1]][1] else 0)&(1 if lm[jointsi_2[0]][1]>lm[jointsi_2[1]][1] else 0)
        self.ma = (1 if lm[jointsm_1[0]][1]>lm[jointsm_1[1]][1] else 0)&(1 if lm[jointsm_2[0]][1]>lm[jointsm_2[1]][1] else 0)
        self.ra = (1 if lm[jointsr_1[0]][1]>lm[jointsr_1[1]][1] else 0)&(1 if lm[jointsr_2[0]][1]>lm[jointsr_2[1]][1] else 0)
        self.pa = (1 if lm[jointsp_1[0]][1]>lm[jointsp_1[1]][1] else 0)&(1 if lm[jointsp_2[0]][1]>lm[jointsp_2[1]][1] else 0)

        #variable to enable writing
        self.ifwrite = [1,1,1,1]
        enable = (sum((self.ia&self.ifwrite[0],self.ma&self.ifwrite[1],self.ra&self.ifwrite[2],self.pa&self.ifwrite[3]))<2)&(~self.iferase)
        self.ifwrite = [enable&i for i in self.ifwrite]


    #Drawing on the image
    def draw_and_erase(self,img, halfside, point):
        for val in self.i:
            for ual in range(len(val)):
                #erase
                if point !=None:
                    if ((val[ual][0]<(point[0]+halfside))&(val[ual][0]>(point[0]-halfside)) \
                    &(val[ual][1]<(point[1]+halfside))&(val[ual][1]>(point[1]-halfside))&self.iferase):
                        self.i.remove(val)
                        break
                #draw
                if ual!=0 & ual!=1:
                    cv2.line(img, val[ual-1], val[ual], color=[255,0,0], thickness = 2)
        for val in self.m:
            for ual in range(len(val)):
                if point !=None:
                    if ((val[ual][0]<(point[0]+halfside))&(val[ual][0]>(point[0]-halfside)) \
                    &(val[ual][1]<(point[1]+halfside))&(val[ual][1]>(point[1]-halfside))&self.iferase):
                        self.m.remove(val)
                        break
                if ual!=0 & ual!=1:
                    cv2.line(img, val[ual-1], val[ual], color=[0,0,255], thickness = 2)
        for val in self.r:
            for ual in range(len(val)):
                if point !=None:
                    if ((val[ual][0]<(point[0]+halfside))&(val[ual][0]>(point[0]-halfside)) \
                    &(val[ual][1]<(point[1]+halfside))&(val[ual][1]>(point[1]-halfside))&self.iferase):
                        self.r.remove(val)
                        break
                if ual!=0 & ual!=1:
                    cv2.line(img, val[ual-1], val[ual], color=[0,255,0], thickness = 2)
        for val in self.p:
            for ual in range(len(val)):
                if point !=None:
                    if ((val[ual][0]<(point[0]+halfside))&(val[ual][0]>(point[0]-halfside)) \
                    &(val[ual][1]<(point[1]+halfside))&(val[ual][1]>(point[1]-halfside))&self.iferase):
                        self.p.remove(val)
                        break
                if ual!=0 & ual!=1:
                    cv2.line(img, val[ual-1], val[ual], color=[0,0,0], thickness = 2)


#Initialization
cap = cv2.VideoCapture(0)
mar = Marker()
marishand = mar.ishand
maria = mar.ia
marma = mar.ma
marra = mar.ra
marpa = mar.pa

#Video feed
while cap.isOpened():
    ret, frame = cap.read()

    #uncomment the below line to get a larger video(same resolution but extrapolated)
    frame = cv2.resize(frame, (1080, 720), interpolation = cv2.INTER_AREA)
    height, width, _ = frame.shape
    hands_lmkf = mar.hands_lmf(frame)
    if mar.ishand==1:
        lmk = mar.lmf2lm(hands_lmkf[0], [height, width])
        mar.is_straight(lmk)
        point = lmk[4]

        #tracing the finger
        if marishand == 0:
            maria = 0
            marma = 0
            marra = 0
            marpa = 0

            if ((mar.ia == 1)&mar.ifwrite[0]):
                mar.i.append([])
                mar.i[-1].append(lmk[ 8])
            if ((mar.ma == 1)&mar.ifwrite[1]):
                mar.m.append([])
                mar.m[-1].append(lmk[12])
            if ((mar.ra == 1)&mar.ifwrite[2]):
                mar.r.append([])
                mar.r[-1].append(lmk[16])
            if ((mar.pa == 1)&mar.ifwrite[3]):
                mar.p.append([])
                mar.p[-1].append(lmk[20])

        if marishand == 1:
            if ((maria==0)&(mar.ia==1)&mar.ifwrite[0]):
                mar.i.append([])
                mar.i[-1].append(lmk[ 8])
            if ((marma==0)&(mar.ma==1)&mar.ifwrite[1]):
                mar.m.append([])
                mar.m[-1].append(lmk[12])
            if ((marra==0)&(mar.ra==1)&mar.ifwrite[2]):
                mar.r.append([])
                mar.r[-1].append(lmk[16])
            if ((marpa==0)&(mar.pa==1)&mar.ifwrite[3]):
                mar.p.append([])
                mar.p[-1].append(lmk[20])
            if ((maria==1)&(mar.ia==1)&mar.ifwrite[0]):
                if len(mar.i)==0:
                    mar.i.append([])
                mar.i[-1].append(lmk[ 8])
            if ((marma==1)&(mar.ma==1)&mar.ifwrite[1]):
                if len(mar.m)==0:
                    mar.m.append([])
                mar.m[-1].append(lmk[12])
            if ((marra==1)&(mar.ra==1)&mar.ifwrite[2]):
                if len(mar.r)==0:
                    mar.r.append([])
                mar.r[-1].append(lmk[16])
            if ((marpa==1)&(mar.pa==1)&mar.ifwrite[3]):
                if len(mar.p)==0:
                    mar.p.append([])
                mar.p[-1].append(lmk[20])
    else:
        point = None
    maria = mar.ia
    marma = mar.ma
    marra = mar.ra
    marpa = mar.pa
    marishand = mar.ishand
    if mar.isflip:
        frame = cv2.flip(frame, 1)
    mar.draw_and_erase(frame, 5, point)
    cv2.imshow("Board", frame)
    if cv2.waitKey(10) & 0xFF == 27:
        break
#termination
cap.release()
cv2.destroyAllWindows()