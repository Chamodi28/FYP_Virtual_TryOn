'''
7

WORKING CODE - ONLY FOR TOPS 

BOTTOM OVERLAY - FAILED

CAMERA DISSAPPERING DUE TO BOTTOM IMAGE UPLOAD ISSUE - UNSOLVED
CORRECT POSITION OF CLOTH - OK

DO NOT CLICK UPLOAD BUTTTON FOR BOTTOMS --> CAMERA WILL BE BLACK
BOTOM OVERLAY CODE IS INCOMPLETE

'''



from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import base64
from rembg import remove
import cvzone

app = Flask(__name__)


apply_status_tops = False
apply_status_bottoms = False

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    cap = cv2.VideoCapture(0)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    
    mp_pose1 = mp.solutions.pose
    pose1 = mp_pose1.Pose()
    
    global apply_status_tops, apply_status_bottoms
    global landmark11x, landmark11y, landmark12x, landmark12y, distance, new_image, logo_resized , x, y, z , zz 
    global multipler, multiplery, shiftxRatio, shiftyRatio, xx, yy, dshiftx, dshifty, dmulx, dmuly , shiftx, shifty
    
    global landmark23x, landmark23y, landmark24x, landmark24y, distance_bottoms, new_image_bottoms, logo_resized_bottoms, x_bottom, y_bottom, z_bottom, zz_bottom
    global multipler_bottom, multiplery_bottom, shiftxRatio_bottom, shiftyRatio_bottom, xx_bottom, yy_bottom, dshiftx_bottom, dshifty_bottom, dmulx_bottom, dmuly_bottom , shiftx_bottom, shifty_bottom
    
    multipler = 1.45    #     1.6   1.7
    multiplery = 1.8   #2.1  2.6   2.1
    shiftxRatio = 0.26 #0.35 0.33  2.3
    shiftyRatio = 0.31 #0.33     0.35      2.9
    x = 2000
    y = 2000
    z = 20
    zz = 20
    
    multipler_bottom = 2.3 #2.3
    multiplery_bottom = 5 # 5
    shiftxRatio_bottom = 0.7 #2.3
    shiftyRatio_bottom = 0.4 #2.9
    x_bottom = 2000
    y_bottom = 2000
    z_bottom = 20
    zz_bottom = 20
    
    
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1) # get mirror view

        if apply_status_tops:
            logo = new_image #cv2.imread('new_image1.png', cv2.IMREAD_UNCHANGED) # new_image1  cloth_image
            logo_resized = cv2.resize(logo, (z,zz)) 
            
        if apply_status_bottoms:
            logo_bottom = new_image_bottoms #cv2.imread('new_image2.png', cv2.IMREAD_UNCHANGED) # new_image1  cloth_image
            logo_resized_bottoms = cv2.resize(logo_bottom, (z_bottom,zz_bottom)) 

        
        if not success:
            break
        else: 
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if apply_status_tops :
                results = pose.process(img_rgb)
                #overlay_logo(img, logo, 10,10)
                
                if results.pose_landmarks:
                    # Draw landmarks on the image
                    mp_drawing = mp.solutions.drawing_utils
                    #mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    for id, lm in enumerate(results.pose_landmarks.landmark):
                        cx, cy = int(lm.x * 640), int(lm.y * 480)  # 1048  640  590  480  resizing landmarkers positions according to window size
                        if id == 11:
                            landmark11x = cx
                            landmark11y = cy

                        if id == 12:
                            landmark12x = cx
                            landmark12y = cy
                            distance = int(((landmark11x - landmark12x) ** 2 + (landmark11y - landmark12y) ** 2) ** 0.5)
                            
                            #cv2.circle(img, (landmark12x, landmark12y), 4, (255, 0, 0), cv2.FILLED)
                            #print(distance)
                            #cv2.circle(img, (landmark12x-100, landmark12y-65), 4, (0, 150, 0), cv2.FILLED)


                    xx = landmark12x
                    yy = landmark12y                   
                    dshiftx = distance
                    dshifty = distance
                    dmulx = distance
                    dmuly = distance
                    shiftx = int(dshiftx*shiftxRatio)
                    shifty = int(dshifty*shiftyRatio)
                    x = landmark12x-shiftx
                    y = landmark12y-shifty
                    z = int(dmulx*multipler)
                    zz = int(dmuly*multiplery)
        
                    img = cvzone.overlayPNG(img, logo_resized, [x, y])
            
            
            if apply_status_bottoms:
                results1 = pose1.process(img_rgb)
                #overlay_logo(img, logo, 10,10)
                
                if results1.pose_landmarks:
                    # Draw landmarks on the image
                    mp_drawing1 = mp.solutions.drawing_utils
                    #mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    for id1, lm1 in enumerate(results1.pose_landmarks.landmark):
                        cx1, cy1 = int(lm1.x * 640), int(lm1.y * 480)  # 1048  640  590  480  resizing landmarkers positions according to window size
                        
                        # right hip
                        if id1 == 23:
                            landmark23x = cx1
                            landmark23y = cy1
                        
                        # left hip
                        if id1 == 24:
                            landmark24x = cx1
                            landmark24y = cy1
                            distance_bottoms = int(((landmark23x - landmark24x) ** 2 + (landmark23y - landmark24y) ** 2) ** 0.5)
                            cv2.circle(img, (landmark24x, landmark24y), 4, (255, 255, 0), cv2.FILLED)
                                 
                    
                    xx_bottom = landmark24x
                    yy_bottom = landmark24y
                    dshiftx_bottom = distance_bottoms
                    dshifty_bottom = distance_bottoms
                    dmulx_bottom = distance_bottoms
                    dmuly_bottom = distance_bottoms
                    shiftx_bottom = int(dshiftx_bottom*shiftxRatio_bottom)
                    shifty_bottom = int(dshifty_bottom*shiftyRatio_bottom)
                    x_bottom = landmark24x-shiftx_bottom
                    y_bottom = landmark24y-shifty_bottom
                    z_bottom = int(dmulx_bottom*multipler_bottom)
                    zz_bottom = int(dmuly_bottom*multiplery_bottom)

                    img = cvzone.overlayPNG(img, logo_resized_bottoms, [x_bottom, y_bottom])

     
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

                

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/upload', methods=['POST'])
def upload():
    global new_image, apply_status_tops, logo_resized
    uploaded_file = request.files['file']
    img_bytes = uploaded_file.read()  
    apply_status_tops = False
          
    
    
    # Perform background removal and overlay
    if img_bytes:
        # Convert bytes to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        image1 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)


        originalImageCopy = image1.copy()
        image2 = cv2.imread('black_bkg.png')
        image3 = cv2.imread('black_bkg.png')
        image4 = cv2.imread('black_bkg.png')
        image5 = cv2.imread('black_bkg.png')


        R = remove(image1, alpha_matting=False)
        image = cvzone.overlayPNG(image2, R, [0, 0])


        # converting to hsv --> define skin color range --> create mask  --> cover skin to black --> remove bg --> overlap
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([5, 48, 80], dtype=np.uint8) # 0, 48, 80
        upper_skin = np.array([20, 255, 255], dtype=np.uint8) # 20, 255, 255
        masks = cv2.inRange(hsv, lower_skin, upper_skin)
        masks = cv2.bitwise_not(masks)
        result = cv2.bitwise_and(image, image, mask=masks) # human skin in black
        RRnotblur = result # remove(result, alpha_matting=True, alpha_matting_background_threshold=37)


        RR = cv2.GaussianBlur(RRnotblur, (19, 19), 0)
        RR = remove(RR)
        RR = cvzone.overlayPNG(image3, RR, [0, 0])


        # gray img --> identify contours --. select largest --> crop editing img and originalCopy to rectangle --> rembg
        gray = cv2.cvtColor(RR, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 15, 250, cv2.THRESH_BINARY) #255
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cloth_cont = max(contours, key=cv2.contourArea)
        cont_image = cv2.drawContours(RR, cloth_cont, -1, (0, 0, 0), 1)


        for y in range(originalImageCopy.shape[0]):
            for x in range(originalImageCopy.shape[1]):
                if cv2.pointPolygonTest(cloth_cont, (x, y), False) >= 0:
                    pixel_value = originalImageCopy[y, x]
                    RR[y, x] = pixel_value
                else:
                    RR[y, x] = [0, 0, 0]


        # gray img --> identify contours --. select largest --> crop editing img and originalCopy to rectangle --> rembg
        gray1 = cv2.cvtColor(RR, cv2.COLOR_BGR2GRAY)
        _, thresh1 = cv2.threshold(gray1, 6, 200, cv2.THRESH_BINARY) #255
        contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cloth_cont1 = max(contours1, key=cv2.contourArea)
        cont_image1 = cv2.drawContours(RR, cloth_cont1, -1, (0, 0, 0), 5)
        x1,y1,w1,h1 = cv2.boundingRect(cloth_cont1)
        cropped_imag1 = cont_image1[y1:y1+h1 , x1:x1+w1]
        

        cropped_imag1 = remove(cropped_imag1)
        hr,wr,cr = cropped_imag1.shape
        image4 = cv2.resize(image4, (wr,hr))
        cropped_imag1 = cvzone.overlayPNG(image4, cropped_imag1, [0, 0])
        cropped_imag1 = remove(cropped_imag1)

      
        
        h_top, w_top, cha = cropped_imag1.shape
        new_width1_top = 768
        new_height1_top = int((768/w_top)*h_top)
    
        new_image = cv2.resize(cropped_imag1, (new_width1_top, new_height1_top))


        # Convert new image to base64 for display
        _, buffer = cv2.imencode('.jpg', new_image)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        apply_status_tops = True
        
        return img_str
    
    else:
        return jsonify({'error': 'No image uploaded'})




@app.route('/uploadBottom', methods=['POST'])
def uploadBottom():
    global new_image_bottoms, apply_status_bottoms, logo_resized_bottoms
    uploaded_file = request.files['file']
    img_bytes = uploaded_file.read()
    
    apply_status_bottoms = False
    
    # Perform background removal and overlay
    if img_bytes:
        # Convert bytes to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        image1 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        image2 = cv2.imread('black_bkg.png')
        image3 = cv2.imread('black_bkg.png')
        image4 = cv2.imread('black_bkg.png')

        R = remove(image1, alpha_matting=False)
        h,w,c = image1.shape
        image2resized = cv2.resize(image2, (w,h))
        image = cvzone.overlayPNG(image2resized, R, [0, 0])


        # converting to hsv --> define skin color range --> create mask  --> cover skin to black --> remove bg --> overlap
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([5, 168, 80], dtype=np.uint8) # 5, 148, 80
        upper_skin = np.array([20, 255, 255], dtype=np.uint8) # 20, 255, 255
        masks = cv2.inRange(hsv, lower_skin, upper_skin)
        masks = cv2.bitwise_not(masks)
        result = cv2.bitwise_and(image, image, mask=masks) # human skin in black

        hb,wb,cb = result.shape
        new_width1 = 768
        new_height1 = int((768/wb)*hb)
        new_image_bottoms = cv2.resize(result, (new_width1, new_height1))
        
        
        # Convert new image to base64 for display
        _, buffer = cv2.imencode('.jpg', new_image_bottoms)
        img_str_bottom = base64.b64encode(buffer).decode('utf-8')
        
        apply_status_bottoms = True
        
        return img_str_bottom
    
    else:
        return jsonify({'error': 'No image uploaded'})





if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000) 



