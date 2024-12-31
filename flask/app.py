from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
from testModelStatic import *
from rembg import remove
import shutil

app = Flask(__name__, template_folder='templates')
app.config['IMAGE_UPLOADS'] = 'static/images'
host = "0.0.0.0"
port = 5000

@app.route("/home", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        image = request.files['file']
        if image.filename == '':
            return redirect("/")
        filename = secure_filename(image.filename)
        basedir = os.path.abspath(os.path.dirname(__file__))
        image.save(os.path.join(basedir, app.config['IMAGE_UPLOADS'], filename))
        return render_template("index.html",filename = filename)
    return render_template("index.html")
    # return "render_template('index.html')"

@app.route("/display/<filename>")
def display(filename):
    return redirect(url_for('static', filename = '/images/'+filename),code=301)

@app.route("/delete/<filename>", methods=["GET"])
def delete(filename):
    basedir = os.path.abspath(os.path.dirname(__file__))
    basedire = os.path.join(basedir, app.config['IMAGE_UPLOADS'])
    file_path = os.path.join(basedire, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return redirect("/")
    else:
        return redirect("/")
    
def find_ear_coordinates(input_image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    image = cv2.imread(input_image)
    h, w, channels=image.shape
    ratio = w / h
    if(w<h):
        if(ratio<1):
            newheight=500
            width=int(ratio*newheight)
            height=int(width/ratio)
            img = cv2.resize(image,(width, newheight))
    elif(w>h):
        if(ratio>1):
            newwidth=500
            height=int(newwidth/ratio)
            img = cv2.resize(image,(newwidth, height))
    else:
        if(w==h):
            img = cv2.resize(image,(400, 400))
    imagedimensions=img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        left_ear_x, left_ear_y = landmarks.part(1).x, landmarks.part(1).y
        right_ear_x, right_ear_y = landmarks.part(15).x, landmarks.part(15).y
        cv2.circle(img, (left_ear_x, left_ear_y), 5, (255, 0, 0), -1)
        cv2.circle(img, (right_ear_x, right_ear_y), 5, (255, 0, 0), -1)
    # cv2.imshow("faceimage",img)
    return left_ear_x, left_ear_y, right_ear_x, right_ear_y, imagedimensions

@app.route('/splitImage', methods=['POST'])
def split():
    try:
        directory_path="static/splitear/"
        files=os.listdir("static/splitear/")
        if((len(files))>0):
            for file in files:
                file_path = os.path.join(directory_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        file_data=request.files
        file_storage = file_data.get('file')
        if file_storage:
            file_name = os.path.splitext(file_storage.filename)[0]
        img = cv2.imdecode(np.frombuffer(request.files['file'].read(), np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 150, 300)
        indices = np.where(edges != [0])
        coordinates = zip(indices[0], indices[1])
        h, w, channels = img.shape
        half2 = w//2
        top = img[:, half2:]
        bottom = img[:, :half2]
        right = remove(top)
        left = remove(bottom)
        cv2.imwrite(f'static/splitear/{file_name}right.png', right)
        cv2.imwrite(f'static/splitear/{file_name}left.png', left)
        return jsonify({"leftear":f"static/splitear/{file_name}left.png","rightear":f"static/splitear/{file_name}right.png"})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == "POST":
            image_data = request.files['file']
            directory_path="static/images/"
            files=os.listdir("static/images/")
            if((len(files))>0):
                for file in files:
                    file_path = os.path.join(directory_path, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
            earrings_path="static/earrings/"
            earringfiles=os.listdir("static/earrings/")
            if((len(earringfiles))>0):
                for ear in earringfiles:
                    ear_path = os.path.join(earrings_path, ear)
                    if os.path.isfile(ear_path):
                        os.remove(ear_path)

            file_path = os.path.join(app.config['IMAGE_UPLOADS'], secure_filename(image_data.filename))
            image_data.save(file_path)
            if(len(os.listdir("static/images/"))==0):
                # print("no file")
                return redirect("/")
            else:
                filename = os.listdir("static/images/")
                image = "static/images/" + str(filename[0])
                detector = FaceShapeDetector(
                    model_path="shape_predictor_81_face_landmarks.dat",
                    shape_predictor_path="shape_predictor_81_face_landmarks.dat",
                    shapemodel_path="data/FaceShapeModel.json"
                )
                face_shape = detector.detect_face_shape(image)
                earringscollection=[]
                if face_shape==None:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    return jsonify({"message": "Please upload file which contain a face"})
                earringscollection=detector.recommend_earrings(face_shape)
                dir_path="static/earrings/"
                earringsdir=[]
                for x in earringscollection[0]:
                    shutil.copy(x,dir_path)
                for z in os.listdir(dir_path):
                    earringsdir.append(z)    
                sortedlist=sorted(earringscollection[1])
                left_ear_x, left_ear_y, right_ear_x, right_ear_y, imagedimensions = find_ear_coordinates(image)
        return jsonify({'filename':str(filename[0]), 'dimension':imagedimensions, 'face_shape':face_shape,'earringsdir':earringsdir,'foldersname':sortedlist, 'leftear':(left_ear_x-38,left_ear_y), 'rightear':(right_ear_x,right_ear_y)}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)})
    

if __name__ == '__main__':
    app.run(debug=True, host=host, port=port)