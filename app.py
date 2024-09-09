import uvicorn
from fastapi import FastAPI
import joblib
from Salary import Salary

app = FastAPI()
joblib_in = open("Salary_Prediction.joblib","rb")
model=joblib.load(joblib_in)


@app.get('/')
def index():
    # return {print(model)}
    return {'message': 'salary Predictionr ML API'}

@app.post('/salary/predict')
def Predict_salary(data:Salary):
    data = data.dict()
    YearsExperience=data['YearsExperience']

    prediction = model.predict([[YearsExperience]])
    
    return {
        'prediction': prediction[0]
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)