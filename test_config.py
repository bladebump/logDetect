from pytorch_lightning.loggers import WandbLogger

# wandb.login(key='7636d0cc1edf410cae67d21d09968d70d6791a89')
wandb_logger = WandbLogger(project="testProject",name="test",save_dir="logs")
config = wandb_logger.experiment.config
config["batch_size"] = 10