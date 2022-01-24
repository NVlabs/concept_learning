NAME=$1
MEMORY=$2
NGPU=1

for CONCEPT in "above45" "abovebb" "near" "upright" "upright45" \
               "alignedvertical" "alignedhorizontal" "front" "front45" "ontop"
do
  for TRAIN_AMT in 100 200 300 400 500 600 700 800 900 1000
  do
    ngc batch run \
      --instance dgx1v.${MEMORY}g.${NGPU}.norm \
      --name $NAME-$CONCEPT-$TRAIN_AMT-lr_0.001_bs_64_adam-model.pointnet \
      --image "nvidian/robotics/weiy_pytorch:1.7.0-py3.7-cuda11.1-cudnn7-devel-ubuntu18.04-egl" \
      --workspace concept_learning:concept_learning \
      --result /result \
      --port 6006 \
      --commandline  "
      export OMP_NUM_THREADS=1
      cd /concept_learning/concept_learning/src
      pip install 'git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib'
      pip install h5py==2.10.0
      pip install moviepy
      pip install open3d --ignore-installed PyYAML
      pip install pytorch3d
      python train/train_oracle_concept.py --concept_dir $CONCEPT \
      --config '/../../configs/pointcloud_oracle.yaml' --train_amt $TRAIN_AMT
      "
  done
done