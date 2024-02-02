# import pytorch_lightning as pl
# import torch
# import torch.nn as nn
# from torchmetrics import MeanSquaredError
#
# class SimpleRegressionModel(pl.LightningModule):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(SimpleRegressionModel, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(input_size, output_size)
#         )
#         self.loss_function = nn.MSELoss()
#         self.mse = MeanSquaredError()
#
#     def forward(self, x):
#         return self.model(x)
#
#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         predictions = self(x)
#         loss = self.loss_function(predictions.squeeze(), y.squeeze())
#         self.log('train_loss', loss.item(), on_epoch=True)
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         y = y.squeeze()
#         predictions = self(x).squeeze()
#         loss = self.loss_function(predictions, y)
#         self.log('val_loss', loss.item(), on_epoch=True)
#         return loss
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
#         return optimizer
