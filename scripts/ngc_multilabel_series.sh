NAME=$1
MEMORY=$2
NGPU=1

ngc batch run \
  --instance dgx1v.${MEMORY}g.${NGPU}.norm \
  --name $NAME--ml-model.pointnet \
  --image "nvcr.io/nvidian/robotics/storm_kit:cuda_10.2_cudnn_driver418.40.04_20210701-154151" \
  --workspace concept_learning:concept_learning \
  --result /result \
  --port 6006 \
  --commandline  "
  export OMP_NUM_THREADS=1
  cd /concept_learning/concept_learning/src
  pip3 install shapely
  pip3 install h5py==2.10.0
  python3 datasets/multilabel_concept_data.py --concept 'above45'
  "