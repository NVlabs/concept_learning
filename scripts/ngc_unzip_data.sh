NAME=$1
MEMORY=$2
NGPU=1

ngc batch run \
  --instance dgx1v.${MEMORY}g.${NGPU}.norm \
  --name $NAME \
  --image "nvidian/robotics/weiy_pytorch:1.7.0-py3.7-cuda11.1-cudnn7-devel-ubuntu18.04-egl" \
  --workspace concept_learning:/concept_learning \
  --result /result \
  --port 6006 \
  --commandline  "
  cd /concept_learning/concept_learning/data
  tar -xf concept_shapenet.tar.xz
  "
