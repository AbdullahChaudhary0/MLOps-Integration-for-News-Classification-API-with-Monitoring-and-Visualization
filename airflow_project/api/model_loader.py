import mlflow.pyfunc

class ModelLoader:
    def __init__(self, model_path: str):
        self.model = mlflow.pyfunc.load_model(model_path)

    def predict(self, texts: list):
        return self.model.predict(texts)
