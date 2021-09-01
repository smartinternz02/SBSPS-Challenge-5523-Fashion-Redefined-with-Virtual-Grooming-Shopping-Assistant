
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask , request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import cv2
from skimage import io
import json
from ibm_watson import AssistantV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

app = Flask(__name__)

model = load_model("fashion.h5")
authenticator = IAMAuthenticator('puP89CTBEWN2-bguNU-co8ESIAcSXgRHl51Uh06_ZPnC')#watson assistant apikey
assistant = AssistantV2(
    version='2021-06-14',
    authenticator = authenticator
)

assistant.set_service_url('https://api.eu-gb.assistant.watson.cloud.ibm.com')#location url
response = assistant.create_session(
        assistant_id='d0cbd1ea-cd4e-4709-bbee-9a7fb3c68f05'#assistant id
    ).get_result()
session_id = response
session_id = session_id["session_id"]
print(type(session_id))
print(session_id)

                 
@app.route('/')
def index():
    print("homepage")
    return render_template('index.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (64,64)) 
        x = image.img_to_array(img)
        #print(x)
        x = np.expand_dims(x,axis =0)
        #print(x)
        preds = model.predict_classes(x)
        print("prediction",preds)
        index = ['dress', 'hat', 'longsleeve', 'outwear', 'pants', 'shirt', 'shoes', 'shorts', 'skirt', 't-shirt']
        text =  str(index[preds[0]])
        
        while True:
            input_text = text
            print(input_text)
            
            response = assistant.message(
                assistant_id='d0cbd1ea-cd4e-4709-bbee-9a7fb3c68f05',
                session_id=session_id,
                input={
                    'message_type': 'text',
                    'text': input_text
                }
            ).get_result()
            print(response)
            recommend_image=(response["output"]["generic"][0]["source"])
            
        
            # Reading an image in default mode
            rec_image = io.imread(recommend_image)
            print(rec_image)
              
            # Window name in which image is displayed
            window_name = 'Recommended image'
              
            # Using cv2.imshow() method 
            # Displaying the image 
            cv2.imshow(window_name, rec_image)
              
            #waits for user to press any key 
            #(this is necessary to avoid Python kernel form crashing)
            cv2.waitKey(0) 
              
            #closing all open windows 
            cv2.destroyAllWindows()
            break
                        
    return text


if __name__ == '__main__':
    app.run(debug = True, threaded = False)
