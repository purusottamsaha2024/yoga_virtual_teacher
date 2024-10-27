RUN for new code:
python app.py


python3 poseLandmark_csv.py -i data/ -o data.csv
python3 create_landmarks.py -i / -o landmarks.csv

python3 poseModel.py -i data.csv -o model.h5
