# earring_recommender_system

# Run frontend

1. cd /client
2. npm install
3. npm run dev


# Run express server

1. cd /server
2. npm install
3. npm run dev

# Run flask server

1. cd /flask
2. python -m venv .venv
## For windows
.venv/Scripts/activate 
## For linux/macOS
.venv/bin/activate

3. pip install -r requirements.txt

# 1. To create model file
python model_formation.py
# 2. earring cluster has already been done
### face_shape dataset link https://www.kaggle.com/datasets/niten19/face-shape-dataset
### earring dataset link https://www.kaggle.com/datasets/asarvazyan/earring-dataset
# 3. Run the server
python app.py
