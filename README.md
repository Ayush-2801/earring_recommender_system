# earring_recommender_system

# Run frontend

cd /client
npm install
npm run dev


# Run express server

cd /server
npm install
npm run dev

# Run flask server

cd /flask
python -m venv .venv
# For windows
.venv/Scripts/activate 
# For linux/macOS
.venv/bin/activate

pip install -r requirements.txt

# 1. To create model file
python model_formation.py
# 2. earring cluster has already been done
# face_shape dataset link https://www.kaggle.com/datasets/niten19/face-shape-dataset
# earring dataset link https://www.kaggle.com/datasets/asarvazyan/earring-dataset
# 3. Run the server
python app.py