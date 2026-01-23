import random
from locust import HttpUser, task, between


class APIUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        # Sample payload reused across requests
        self.payload = {
            "age": random.randint(18, 60),
            "gender": random.randint(0, 1),
            "handedness": random.randint(0, 1),
            "verbal_iq": random.randint(80, 130),
            "performance_iq": random.randint(80, 130),
            "full4_iq": random.uniform(80, 130),
            "adhd_index": random.uniform(0, 1),
            "inattentive": random.uniform(0, 1),
            "hyper_impulsive": random.uniform(0, 1),
        }

    @task(2)
    def health_check(self):
        self.client.get("/")

    @task(1)
    def drift_check(self):
        self.client.get("/drift", params={"n": 200, "psi_threshold": 0.2}, name="/drift")

    @task(3)
    def predict(self):
        n_clusters = random.choice([0, 1, 3, 5])
        self.client.post(f"/predict?n_clusters={n_clusters}", json=self.payload, name="/predict")
