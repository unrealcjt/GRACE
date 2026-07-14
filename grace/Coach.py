import copy
import time
from comet_ml import Experiment, Optimizer
import numpy as np
from numpy.core import overrides
import torch
from tqdm import tqdm
from sklearn import metrics
import grace
from thop import profile

log = grace.utils.get_logger()


def get_inference_time(model, data):
  model = model.cuda()
  model.eval()

  # 2. 预热 (Warm-up) —— 至少 10-20 次
  # 消除 CUDA 显存首次分配和算子初始化的影响
  with torch.no_grad():
      for _ in range(20):
          _ = model(data)

  # 3. 创建 CUDA Event 专门用于 GPU 测时
  start_event = torch.cuda.Event(enable_timing=True)
  end_event = torch.cuda.Event(enable_timing=True)

  repetitions = 200 # 循环运行 200 次求平均值
  start_event.record()

  with torch.no_grad():
      for _ in range(repetitions):
          _ = model(data)

  end_event.record()

  # 4. 关键：同步 GPU，等待所有计算完成
  torch.cuda.synchronize()

  # 5. 计算单次推理的平均时间（单位：毫秒 ms）
  total_time_ms = start_event.elapsed_time(end_event)
  avg_time_ms = total_time_ms / repetitions

  print(f"Batch Size = 1 时的单次推理时间: {avg_time_ms:.2f} ms")


class Coach:
    def __init__(self, trainset, devset, testset, model, opt, sched, args):
        self.trainset = trainset
        self.devset = devset
        self.testset = testset
        self.model = model
        self.opt = opt
        self.scheduler = sched
        self.args = args
        if args.log_in_comet:
            self.experiment = grace.Logger()

        self.dataset_label_dict = {
            "iemocap": {"hap": 0, "sad": 1, "neu": 2, "ang": 3, "exc": 4, "fru": 5},
            "iemocap_4": {"hap": 0, "sad": 1, "neu": 2, "ang": 3},
            "mosei": {"Negative": 0, "Positive": 1},
        }

        if args.emotion == "7class":
            self.label_to_idx = {
                "Strong Negative": 0,
                "Weak Negative": 1,
                "Negative": 2,
                "Neutral": 3,
                "Positive": 4,
                "Weak Positive": 5,
                "Strong Positive": 6,
            }
        else:
            self.label_to_idx = self.dataset_label_dict[args.dataset]

        self.best_dev_f1 = None
        self.best_epoch = None
        self.best_state = None

    def load_ckpt(self, ckpt):
        self.best_dev_f1 = ckpt["best_dev_f1"]
        self.best_epoch = ckpt["best_epoch"]
        self.best_state = ckpt["best_state"]
        self.model.load_state_dict(self.best_state)
        print("Loaded model.....")

    def train(self):
        log.debug(self.model)
        # Early stopping.
        best_dev_f1, best_epoch, best_state = (
            self.best_dev_f1,
            self.best_epoch,
            self.best_state,
        )

        dev_f1s = []
        test_f1s = []
        train_losses = []
        best_test_f1 = None

        # Train
        for epoch in range(1, self.args.epochs + 1):
            train_loss = self.train_epoch(epoch)
            dev_f1, dev_loss = self.evaluate()
            self.scheduler.step(dev_loss)
            test_f1, _ = self.evaluate(test=True)
    
            log.info("[Dev set] [f1 {:.4f}]".format(dev_f1))
            if best_dev_f1 is None or dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_test_f1 = test_f1
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                if self.args.dataset == "mosei":
                    torch.save(
                        {"args": self.args, "state_dict": self.model},
                        self.args.data_root+
                        "/model_checkpoints/mosei_best_dev_f1_model_"
                        + self.args.modalities
                        + "_"
                        + self.args.emotion
                        + ".pt",
                    )
                else:
                    torch.save(
                        {"args": self.args, "state_dict": self.model},
                        self.args.data_root
                        + "/model_checkpoints/"
                        + self.args.dataset
                        + "_best_dev_f1_model_"
                        + self.args.modalities
                        + ".pt",
                    )

                log.info("Save the best model.")
            log.info("[Test set] [f1 {:.4f}]".format(test_f1))

            dev_f1s.append(dev_f1)
            test_f1s.append(test_f1)
            train_losses.append(train_loss)

            if self.args.log_in_comet:
                self.experiment.log_metric("F1 Score (Dev)", dev_f1, epoch=epoch)
                self.experiment.log_metric("F1 Score (test)", test_f1, epoch=epoch)
                self.experiment.log_metric("train_loss", train_loss, epoch=epoch)
                self.experiment.log_metric("val_loss", dev_loss, epoch=epoch)

        # The best

        self.model.load_state_dict(best_state)
        log.info("")
        log.info("Best in epoch {}:".format(best_epoch))
        dev_f1, _ = self.evaluate()
        log.info("[Dev set] [f1 {:.4f}]".format(dev_f1))
        test_f1, _ = self.evaluate(test=True)
        log.info("[Test set] f1 {}".format(test_f1))
        if (self.args.log_in_comet):
            self.experiment.log_metric("best_dev_f1", best_dev_f1, epoch=epoch)
            self.experiment.log_metric("best_test_f1", best_test_f1, epoch=epoch)

        return best_dev_f1, best_epoch, best_state, train_losses, dev_f1s, test_f1s

    def train_epoch(self, epoch):
        start_time = time.time()
        epoch_loss = 0
        self.model.train()

        self.trainset.shuffle()
        for idx in tqdm(range(len(self.trainset)), desc="train epoch {}".format(epoch)):
            self.model.zero_grad()
            data = self.trainset[idx]
            for k, v in data.items():
                if not k == "utterance_texts":
                    data[k] = v.to(self.args.device)

            # macs, params = profile(self.model, inputs=(data, ))
            # print(f"Flops (M): {macs * 2 / 1e6}, Params (M): {params / 1e6}")
            # get_inference_time(self.model, data)
            # return
        
            nll = self.model.get_loss(data)
            epoch_loss += nll.item()
            nll.backward()
            self.opt.step()

        end_time = time.time()
        log.info("")
        log.info(
            "[Epoch %d] [Loss: %f] [Time: %f]"
            % (epoch, epoch_loss, end_time - start_time)
        )
        return epoch_loss

    def evaluate(self, test=False):
        dev_loss = 0
        dataset = self.testset if test else self.devset
        self.model.eval()
        with torch.no_grad():
            golds = []
            preds = []
            for idx in tqdm(range(len(dataset)), desc="test" if test else "dev"):
                data = dataset[idx]
                golds.append(data["label_tensor"])
                for k, v in data.items():
                    if not k == "utterance_texts":
                        data[k] = v.to(self.args.device)
                y_hat = self.model(data)
                preds.append(y_hat.detach().to("cpu"))
                nll = self.model.get_loss(data)
                dev_loss += nll.item()

            
            golds = torch.cat(golds, dim=-1).numpy()
            preds = torch.cat(preds, dim=-1).numpy()
            f1 = metrics.f1_score(golds, preds, average="weighted")

            if test:
                print(
                    metrics.classification_report(
                        golds, preds, target_names=self.label_to_idx.keys(), digits=4
                    )
                )



                if self.args.log_in_comet:
                    self.experiment.log_confusion_matrix(
                        golds,
                        preds,
                        labels=list(self.label_to_idx.keys()),
                        overwrite=True,
                    )

        return f1, dev_loss

# import copy
# import time
# from comet_ml import Experiment, Optimizer
# import numpy as np
# from numpy.core import overrides
# import torch
# from tqdm import tqdm
# from sklearn import metrics
# import grace

# log = grace.utils.get_logger()


# class Coach:
#     def __init__(self, trainset, devset, testset, model, opt, sched, args):
#         self.trainset = trainset
#         self.devset = devset
#         self.testset = testset
#         self.model = model
#         self.opt = opt
#         self.scheduler = sched
#         self.args = args
#         if args.log_in_comet:
#             self.experiment = grace.Logger()

#         self.dataset_label_dict = {
#             "iemocap": {"hap": 0, "sad": 1, "neu": 2, "ang": 3, "exc": 4, "fru": 5},
#             "iemocap_4": {"hap": 0, "sad": 1, "neu": 2, "ang": 3},
#             "mosei": {"Negative": 0, "Positive": 1},
#         }

#         if args.emotion == "7class":
#             self.label_to_idx = {
#                 "Strong Negative": 0,
#                 "Weak Negative": 1,
#                 "Negative": 2,
#                 "Neutral": 3,
#                 "Positive": 4,
#                 "Weak Positive": 5,
#                 "Strong Positive": 6,
#             }
#         else:
#             self.label_to_idx = self.dataset_label_dict[args.dataset]

#         self.best_dev_f1 = None
#         self.best_epoch = None
#         self.best_state = None

#     def load_ckpt(self, ckpt):
#         self.best_dev_f1 = ckpt["best_dev_f1"]
#         self.best_epoch = ckpt["best_epoch"]
#         self.best_state = ckpt["best_state"]
#         self.model.load_state_dict(self.best_state)
#         print("Loaded model.....")

#     def train(self):
#         log.debug(self.model)
#         # Early stopping.
#         best_dev_f1, best_epoch, best_state = (
#             self.best_dev_f1,
#             self.best_epoch,
#             self.best_state,
#         )

#         dev_accs = []
#         test_accs = []
#         train_losses = []
#         best_test_f1 = None
#         best_dev_acc = None

#         # Train
#         for epoch in range(1, self.args.epochs + 1):
#             train_loss = self.train_epoch(epoch)
#             dev_acc, dev_loss = self.evaluate()
#             self.scheduler.step(dev_loss)
#             test_acc, _ = self.evaluate(test=True)

#             log.info("[Dev set] [acc {:.4f}]".format(dev_acc))
#             if best_dev_acc is None or dev_acc > best_dev_acc:
#                 best_dev_acc = dev_acc
#                 best_test_acc = test_acc
#                 best_epoch = epoch
#                 best_state = copy.deepcopy(self.model.state_dict())
#                 if self.args.dataset == "mosei":
#                     torch.save(
#                         {"args": self.args, "state_dict": self.model},
#                         self.args.data_root +
#                         "/model_checkpoints/mosei_best_dev_acc_model_"
#                         + self.args.modalities
#                         + "_"
#                         + self.args.emotion
#                         + ".pt",
#                     )
#                 else:
#                     torch.save(
#                         {"args": self.args, "state_dict": self.model},
#                         self.args.data_root
#                         + "/model_checkpoints/"
#                         + self.args.dataset
#                         + "_best_dev_acc_model_"
#                         + self.args.modalities
#                         + ".pt",
#                     )

#                 log.info("Save the best model.")
#             log.info("[Test set] [acc {:.4f}]".format(test_acc))

#             dev_accs.append(dev_acc)
#             test_accs.append(test_acc)
#             train_losses.append(train_loss)

#             if self.args.log_in_comet:
#                 self.experiment.log_metric("Acc Score (Dev)", dev_acc, epoch=epoch)
#                 self.experiment.log_metric("Acc Score (test)", test_acc, epoch=epoch)
#                 self.experiment.log_metric("train_loss", train_loss, epoch=epoch)
#                 self.experiment.log_metric("val_loss", dev_loss, epoch=epoch)

#         # The best

#         self.model.load_state_dict(best_state)
#         log.info("")
#         log.info("Best in epoch {}:".format(best_epoch))
#         dev_acc, _ = self.evaluate()
#         log.info("[Dev set] [acc {:.4f}]".format(dev_acc))
#         test_acc, _ = self.evaluate(test=True)
#         log.info("[Test set] acc {}".format(test_acc))
#         if (self.args.log_in_comet):
#             self.experiment.log_metric("best_dev_acc", best_dev_acc, epoch=epoch)
#             self.experiment.log_metric("best_test_acc", best_test_acc, epoch=epoch)

#         return best_dev_f1, best_epoch, best_state, train_losses, dev_accs, test_accs

#     def train_epoch(self, epoch):
#         start_time = time.time()
#         epoch_loss = 0
#         self.model.train()

#         self.trainset.shuffle()
#         for idx in tqdm(range(len(self.trainset)), desc="train epoch {}".format(epoch)):
#             self.model.zero_grad()
#             data = self.trainset[idx]
#             for k, v in data.items():
#                 if not k == "utterance_texts":
#                     data[k] = v.to(self.args.device)

#             nll = self.model.get_loss(data)
#             epoch_loss += nll.item()
#             nll.backward()
#             self.opt.step()

#         end_time = time.time()
#         log.info("")
#         log.info(
#             "[Epoch %d] [Loss: %f] [Time: %f]"
#             % (epoch, epoch_loss, end_time - start_time)
#         )
#         return epoch_loss

#     def evaluate(self, test=False):
#         dev_loss = 0
#         dataset = self.testset if test else self.devset
#         self.model.eval()
#         with torch.no_grad():
#             golds = []
#             preds = []
#             for idx in tqdm(range(len(dataset)), desc="test" if test else "dev"):
#                 data = dataset[idx]
#                 golds.append(data["label_tensor"])
#                 for k, v in data.items():
#                     if not k == "utterance_texts":
#                         data[k] = v.to(self.args.device)
#                 y_hat = self.model(data)
#                 preds.append(y_hat.detach().to("cpu"))
#                 nll = self.model.get_loss(data)
#                 dev_loss += nll.item()

#             golds = torch.cat(golds, dim=-1).numpy()
#             preds = torch.cat(preds, dim=-1).numpy()
#             acc = metrics.accuracy_score(golds, preds)

#             if test:
#                 print(
#                     metrics.classification_report(
#                         golds, preds, target_names=self.label_to_idx.keys(), digits=4
#                     )
#                 )

#                 if self.args.log_in_comet:
#                     self.experiment.log_confusion_matrix(
#                         golds,
#                         preds,
#                         labels=list(self.label_to_idx.keys()),
#                         overwrite=True,
#                     )

#         return acc, dev_loss