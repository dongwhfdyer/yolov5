cp kuhn_utils/.vimrc ~/.vimrc
source ~/.vimrc
apt-get update
apt-get install -y tmux vifm rar

rar x datasets/NEU-DET.rar
cp -r NEU-DET/TRAIN datasets/NEU-DET_
rm -rf NEU-DET
mv datasets/NEU-DET_ datasets/neu_det
python kuhn_utils.py
pip install pyyaml seaborn wandb
pip install --upgrade tensorboard

