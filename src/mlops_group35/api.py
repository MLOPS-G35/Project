from fastapi import FastAPI
from http import HTTPStatus
from hydra import initialize, compose

from pydantic import BaseModel

from mlops_group35.cluster_train import build_train_config
from mlops_group35.data import load_csv_for_clustering


class PredictionInput(BaseModel):
    age: int
    gender: int
    handedness: int
    verbal_iq: int
    performance_iq: int
    full4_iq: float
    adhd_index: float
    inattentive: float
    hyper_impulsive: float


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
    field_names = data.model_fields.keys()


    # data_dict = {
    #     "Age": data.Age,
    #     "Gender": data.Gender,
    #     "Handedness": data.Handedness,
    #     "Verbal IQ": data.Verbal_IQ,
    #     "Performance IQ": data.Performance_IQ,
    #     "Full4 IQ": data.Full4_IQ,
    #     "ADHD Index": data.ADHD_Index,
    #     "Inattentive": data.Inattentive,
    #     "Hyper/Impulsive": data.Hyper_Impulsive,
    # }



    # with initialize(config_path="../../configs", version_base="1.3"):
    #     cfg = compose(config_name="cluster")
    #
    # train_cfg = build_train_config(cfg)

    csv_path = "data/processed/combined.csv"
    id_col = "scandir_id"
    ids, df = load_csv_for_clustering(csv_path, id_col, field_names)

    print(df.tail(5))

    return {"features_used"}
