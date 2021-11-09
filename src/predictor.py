import csv
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from src.dataset import get_dataloader
from src.models import get_model
from src.utils.visualize import Visualizer

class DefaultPredictor(object):

    def __init__(self, cfg):

        self.cfg = cfg
        self.model = get_model(cfg)
        state_dict = torch.load(self.cfg.MODEL.WEIGHTS)
        self.model.load_state_dict(state_dict)
    
        self.test_dir_p = (Path(self.cfg.OUTPUT_DIR) / "submission")
        self.test_dir_p.mkdir(parents=True, exist_ok=True)

        self.test_csv_p = (Path(self.cfg.OUTPUT_DIR) / "submission.csv")
    
    def test(self):

        test_dataloader = get_dataloader(self.cfg, 'test')

        self.model = self.model.cuda()
        self.model.eval()

        csvfile = open(str(self.test_csv_p), 'w', newline='\n')
        writer = csv.writer(csvfile)
        writer.writerow(["Id", "Predicted"])
        for image, image_id in tqdm(test_dataloader, desc='Predict'):

            with torch.no_grad():
                image = image.cuda()
                output = self.model(image)
                output = F.softmax(output, dim=1)
                pred = torch.argmax(output, dim=1)
            
            B, _, _ = pred.shape
            for b in range(B):
                vis_image_path = str(self.test_dir_p / f"{image_id[b]}.png")
                Visualizer.save_multi_channel_as_png(image[b:b+1,:,:,:].detach().cpu(), vis_image_path)
                vis_pred_path = str(self.test_dir_p / f"{image_id[b]}_pred.png")
                Visualizer.save_multi_channel_as_png(output[b:b+1,:,:,:].detach().cpu(), vis_pred_path)
                label = pred[b,:,:]
                H, W = label.shape
                for i in range(H):
                    for j in range(W):
                        key = f"{image_id[b]}_{i:03d}_{j:03d}"
                        value = label[i,j].item()
                        writer.writerow([key, value])
        csvfile.close()
        
        