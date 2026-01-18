from fastapi import FastAPI
from http import HTTPStatus

from pydantic import BaseModel
class PredictionInput(BaseModel):
    Age: int
    Gender: int
    Handedness: int
    Verbal_IQ: int
    Performance_IQ: int
    Full4_IQ: float
    ADHD_Index: float
    Inattentive: float
    Hyper_Impulsive: float


app = FastAPI()



@app.get("/")
def root():
    """ Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}


@app.post("/predict")
def predict(data: PredictionInput):
    data_dict = {
        "Age": data.Age,
        "Gender": data.Gender,
        "Handedness": data.Handedness,
        "Verbal IQ": data.Verbal_IQ,
        "Performance IQ": data.Performance_IQ,
        "Full4 IQ": data.Full4_IQ,
        "ADHD Index": data.ADHD_Index,
        "Inattentive": data.Inattentive,
        "Hyper/Impulsive": data.Hyper_Impulsive,
    }



    return {"features_used"}
