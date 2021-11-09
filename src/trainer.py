import logging
import pprint
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from src.losses import get_loss
from src.dataset import get_dataloader
from src.models import get_model
from src.optimizer import get_optim_and_scheduler
from src.utils.visualize import Visualizer


class DefaultTrainer(object):

    def __init__(self, cfg):

        self.cfg = cfg
        self.model = get_model(cfg)

        if self.cfg.TRAINER.RESUME:
            state_dict = torch.load(self.cfg.MODEL.WEIGHTS)
            self.model.load_state_dict(state_dict)

        self.global_iter = 0

        self.checkpoint_path_p = (Path(self.cfg.OUTPUT_DIR) / self.cfg.CHECKPOINT_PATH)

        self.write_dir_p = (Path(self.cfg.OUTPUT_DIR) / self.cfg.SUMMARY_DIR)
        self.write_dir_p.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.write_dir_p))

        self.vis_dir_p = (Path(self.cfg.OUTPUT_DIR) / self.cfg.VIS_DIR)
        self.vis_dir_p.mkdir(parents=True, exist_ok=True)
    
    def __del__(self):

        self.writer.close()
    
    def write(self, loss_dict, cur_iter=None):

        if cur_iter is None:
            cur_iter = self.global_iter

        for k, v in loss_dict.items():
            self.writer.add_scalar(k, v, cur_iter)

    def train(self):

        train_dataloader = get_dataloader(self.cfg, 'train')
        optim, scheduler = get_optim_and_scheduler(self.cfg, self.model.parameters())
        criterion = get_loss(self.cfg)

        self.model = self.model.cuda()
        self.model.train()
        MAX_EPOCH = (self.cfg.TRAINER.MAX_ITER // len(train_dataloader)) + 1
        logging.info(f"Start {MAX_EPOCH} epoch training...")
        for epoch in range(MAX_EPOCH):

            for image, label in train_dataloader:

                image = image.cuda()
                label = label.cuda()
                output = self.model(image)
                output = F.softmax(output, dim=1)
                mask_index = torch.sum(label, dim=(-1, -2)) > 0
                mask_index[:,0] = False
                loss = criterion(output[mask_index.bool(),:,:], label[mask_index.bool(),:,:])

                optim.zero_grad()
                loss.backward()
                optim.step()
                scheduler.step()

                if self.global_iter % self.cfg.TRAINER.PRINT_ITER == 0:

                    loss_dict = dict(
                        loss_all=loss,
                    )

                    msg = f"[Epoch {epoch}/{MAX_EPOCH}] [Iter {self.global_iter}/{self.cfg.TRAINER.MAX_ITER}]"
                    msg += f" {pprint.pformat(loss_dict)}"
                    logging.info(msg)

                    self.write(loss_dict, self.global_iter)

                    image_path = str(self.vis_dir_p / f"iter_{self.global_iter}_input.png")
                    image_vis = image[0:1].detach().cpu()
                    Visualizer.save_multi_channel_as_png(image_vis, image_path)
                    output_path = str(self.vis_dir_p / f"iter_{self.global_iter}_output.png")
                    Visualizer.save_multi_channel_as_png(output.detach().cpu(), output_path)
                    label_path = str(self.vis_dir_p / f"iter_{self.global_iter}_label.png")
                    Visualizer.save_multi_channel_as_png(label.detach().cpu(), label_path)

                    torch.save(self.model.state_dict(), str(self.checkpoint_path_p))

                self.global_iter += 1
                if self.global_iter > self.cfg.TRAINER.MAX_ITER:
                    break

    def valid(self):

        valid_dataloader = get_dataloader(self.cfg, 'valid')

        self.model = self.model.cuda()
        self.model.eval()
        correct_all = []
        correct_ratio_all = []
        batch_cnt = 0
        for image, label in tqdm(valid_dataloader, desc='Validation'):

            image = image.cuda()
            label = label.cuda()
            with torch.no_grad():
                output = self.model(image)
                output = F.softmax(output, dim=1)
                pred = torch.argmax(output, dim=1)
                target = torch.argmax(label, dim=1)
                correct = (pred == target)
                B, H, W = correct.shape
                correct_ratio = torch.sum(correct, dim=(-1,-2)) / (H * W)
            correct_all.append(correct)
            correct_ratio_all.append(correct_ratio)
            batch_cnt += B
        
        correct_all = torch.cat(correct_all, dim=0)
        torch.save(correct_all.detach().cpu(), str(Path(self.cfg.OUTPUT_DIR) / "correct.pth"))

        correct_ratio_one = torch.sum(torch.cat(correct_ratio_all, dim=0)) / batch_cnt
        logging.info(f"Pixel Accuracy: {correct_ratio_one}")
