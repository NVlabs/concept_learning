NAME=$1
MEMORY=$2
NGPU=1

for CONCEPT in "above45" "abovebb" "near" "upright" "upright45" \
               "alignedvertical" "alignedhorizontal" "front" "front45" "ontop"
do
  ngc batch run \
    --instance dgx1v.${MEMORY}g.${NGPU}.norm.beta \
    --name $NAME-$CONCEPT-$TRAIN_AMT-lr_0.01_bs_32_adam_model.MLP \
    --image "nvcr.io/nvidian/robotics/storm_kit:cuda_10.2_cudnn_driver418.40.04_20210701-154151" \
    --workspace concept_learning:concept_learning \
    --result /result \
    --port 6006 \
    --commandline  "
    export OMP_NUM_THREADS=1
    cd /concept_learning/concept_learning/src
    pip3 install 'git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib'
    pip3 install shapely
    pip3 install open3d-python
    pip3 install h5py==2.10.0
    pip3 install moviepy
    pip3 install pytorch3d
    python3 datasets/collect_human_data.py --config '/../../configs/rawstate_AL.yaml' --concept $CONCEPT \
    --simulated --active_samples 1000 --passive_samples 0 --batch_size 100 --objective 'confusion' \
    --warmstart 0 --mining 0
    "
done