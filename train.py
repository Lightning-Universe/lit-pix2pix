# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from src.dataset import FacadesDataset
from torch.utils.data import DataLoader
from src.pix2pix import Pix2Pix  
import lightning as L

train_dataset = FacadesDataset(os.getcwd() + '/facades/train', target_size=256)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

val_dataset = FacadesDataset(os.getcwd() + '/facades/val', target_size=256)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)

pix2pix = Pix2Pix(3, 3)
trainer = L.Trainer(max_epochs=20, fast_dev_run=True)
trainer.fit(pix2pix, train_dataloader, val_dataloader)