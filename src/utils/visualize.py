import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib import cm
from torchvision.utils import make_grid


def handle_3d_image_tensor(data, ax):
    data = data - data.min()
    data = data / data.max()
    data_vis = data.permute(1, 2, 0)
    ax.imshow(data_vis)

def handle_4d_image_tensor(data, ax):
    data = data - data.min()
    data = data / data.max()
    data_vis = data[0].permute(1, 2, 0)
    ax.imshow(data_vis)

def handle_image(data, ax):
    data = data - data.min()
    data = data / data.max()
    ax.imshow(data)

def handle_landmark(data, ax):
    B, N, _ = data.shape
    color_list = cm.viridis(np.linspace(0.0, 1.0, N))
    for b in range(B):
        ax.scatter(data[b, :, 0], data[b, :, 1], s=3, c=color_list)

TYPE_HANDLER = dict(
    image=handle_image,
    image_tensor_chw=handle_3d_image_tensor,
    image_tensor_bchw=handle_4d_image_tensor,
    landmark2d=handle_landmark,
)

class Visualizer:

    def __init__(self,
                 save_path,
                 grid=(1,1),
                 axis='off'):

        self.save_path = save_path
        self.row, self.column = grid

        self.fig, self.axs = plt.subplots(
            self.row,
            self.column,
            squeeze=False,
            figsize=(self.column * 3.0, self.row * 4.0)
        )
        self.fig.tight_layout()
        if axis == 'off':
            for i in range(self.row):
                for j in range(self.column):
                    self.axs[i, j].axis('off')
    
    def draw(self, data, type, row=1, column=1):

        if type not in TYPE_HANDLER.keys():
            raise ValueError(f"Type {type} is not in {TYPE_HANDLER.keys()}")
        
        ax = self.axs[row - 1, column - 1]
        TYPE_HANDLER[type](data, ax)
    
    def save(self):
        plt.savefig(self.save_path)
        plt.close()
    
    @classmethod
    def draw_and_save(cls, data, type, vis_path, axis='off'):
        if axis == 'off':
            plt.axis('off')
        if type not in TYPE_HANDLER.keys():
            raise ValueError(f"Type {type} is not in {TYPE_HANDLER.keys()}")
        ax = plt.gca()
        plt.tight_layout()
        TYPE_HANDLER[type](data, ax)
        plt.savefig(vis_path)
        plt.close()
    
    @classmethod
    def save_image_as_png(cls, data, vis_path):
        data = data - data.min()
        data = data / data.max() * 255.0
        data = data[0].permute(1, 2, 0).numpy()[:,:,::-1]
        cv2.imwrite(vis_path, data)
    
    @classmethod
    def save_multi_channel_as_png(cls, data, vis_path):

        data = data[0,:,None,:,:]
        data_vis = make_grid(data)
        data_vis = data_vis - data_vis.min()
        data_vis = data_vis / data_vis.max() * 255.0
        data_vis = data_vis.permute(1, 2, 0).numpy()[:,:,::-1]
        cv2.imwrite(vis_path, data_vis)
