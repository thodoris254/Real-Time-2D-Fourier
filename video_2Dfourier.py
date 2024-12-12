import cv2
import numpy as np
import time # can be removed
from numpy import fft

LP = 0
HP = 1

# --- input
vid = cv2.VideoCapture(0) # set to 0 to capture 1st camera, 1 for 2nd camera if connected etc

a = 1/7 # ratio of pass filter / frame | put 1 for no filtering
c = HP # LP/HP, switches square to be black-white and white-black

# good combinations are a = 1/4, c = LP | a = 1/7, c = HP 
#


cTime = 0
pTime = 0

_, filter = vid.read()

print("\nSquare is set to: 1/{:d} of frame".format(int(1/a)))
if c==HP: print("filter: High Pass")
elif c==LP: print("filter: Low Pass")
print("Press 'q' t exit.\n")

filter = cv2.cvtColor(filter, cv2.COLOR_BGR2GRAY)
shape_x = filter.shape[0]
shape_y = filter.shape[1]

filter[::,::] = abs(c) # creating all zeros or all ones frame/filter

filter[int((1-a)*shape_x/2):int((1+a)*shape_x/2):,int((1-a)*shape_y/2):int((1+a)*shape_y/2):] = abs(c-1) # setting filter all 1 on inside square and 0 on outside for LowPass. Inverse for HighPass


while True:
    _, frame = vid.read()
    frame = frame[::,::-1] # mirroring image

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # opencv uses BGR, not RGB

    fft_frame = fft.fftshift(fft.fft2(frame)) # getting 2D fourier and shifting dc to middle
    fft_frame_db = 20*np.log(np.abs(1+fft_frame)) # turning in db
    
    fft_filtered_frame = np.multiply(fft_frame_db,filter) # multiplying bit by bit with filter, either by 1 or 0

    fft_frame_show = np.asarray(np.abs(fft_filtered_frame), dtype=np.uint8) # changing type so it can be recognized by cv2.imshow()

    fft_filtered_frame = np.exp(fft_filtered_frame/20)-1 # changing from db to original
    filtered_frame = fft.ifft2(np.abs(fft_filtered_frame)*np.exp(1j*np.angle(fft_frame))) # using inverse fft on abs*exp(angle)

    filtered_frame_show = np.asarray(np.abs(filtered_frame), dtype=np.uint8) # changing type so it can be recognized by cv2.imshow()

    # \/ for fps
    cTime = time.time() 
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(fft_frame_show,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3, (51,222,32), 3) # for display
    # /\

    cv2.imshow("Fourier 2D", fft_frame_show) # showing filtered fft
    cv2.imshow("camera", filtered_frame_show) # showing filtered video
    
    key = cv2.waitKey(1)
    if key ==ord("q"): break # pess q to exit


vid.release()
cv2.destroyAllWindows()