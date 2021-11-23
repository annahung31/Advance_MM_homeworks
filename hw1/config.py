from easydict import EasyDict as edict

config = edict()
config.dataset = "APD"
config.embedding_size = 512
config.sample_rate = 1
config.fp16 = False
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.1  # batch size is 512
config.output = "./models/0415-1222"

if config.dataset == "APD":
    config.rec = "data_utils"
    config.num_classes = 685
    config.num_image = "forget"
    config.num_epoch = 200
    config.warmup_epoch = -1
    config.val_targets = ["val"]


    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
            [m for m in [20, 28, 32] if m - 1 <= epoch])
    config.lr_func = lr_step_func

