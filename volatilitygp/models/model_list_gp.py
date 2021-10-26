from botorch.models import ModelListGP as _ModelListGP


class ModelListGP_N(_ModelListGP):
    @property
    def batch_shape(self):
        return self.models[0].batch_shape
