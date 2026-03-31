

import numpy as np
import torch
from torch.utils.data import Dataset

from ik.data.pipeline import load_split


class IKDataset(Dataset):


  def __init__(
      self,

      split: str,
      save_dir: str,
      MinMax_X: list | None = None,
      MinMax_Y: list | None = None,
  ):

    q_init,q_target,P_target,R6_target=load_split(save_dir,split)
    R6_target=R6_target.reshape(-1,6)  # (n,2,3) -> (n,6)

    y=q_target.copy()
    X=np.concatenate([R6_target,P_target,q_init],axis=1)
    if MinMax_X is None:
      self.MinMax_Y=[y.min(axis=0),y.max(axis=0)]
      self.MinMax_X=[X.min(axis=0),X.max(axis=0)]
    else:
      self.MinMax_Y=MinMax_Y
      self.MinMax_X=MinMax_X

    self.y=(y-self.MinMax_Y[0])/(self.MinMax_Y[1]-self.MinMax_Y[0])
    self.y=(self.y*2)-1
    self.X=(X-self.MinMax_X[0])/(self.MinMax_X[1]-self.MinMax_X[0])
    self.X=(self.X*2)-1
    self.q_init=q_init
    self.q_target=q_target
    self.P_target=P_target
    self.R6_target=R6_target




  def __len__(self) -> int:
      return len(self.X)

  def __getitem__(self, idx):
      return (
          self.X[idx],
          self.y[idx],
          self.q_init[idx],
          self.q_target[idx],
          self.P_target[idx],
          self.R6_target[idx],
      )