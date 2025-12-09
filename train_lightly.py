import lightly_train
from models.backbone import BackboneModel


model = BackboneModel('cfg/training/yolov7.yaml')
if __name__ == "__main__":
    lightly_train.train(
        out="out/my_experiment", 
        data="data",
        model=model,
        method="distillationv2",
        method_args={
            "teacher": "dinov3/vitb16",
        }
        epochs=10
    )