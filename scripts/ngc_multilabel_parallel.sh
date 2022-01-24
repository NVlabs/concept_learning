NAME=$1
MEMORY=$2
NGPU=1

for CONCEPT in "above45" "abovebb" "near" "upright" "upright45" \
               "alignedvertical" "alignedhorizontal" "front" "front45" "ontop"
do
  for TRAIN_AMT in 100 200 300 400 500 600 700 800 900 1000
  do
    ngc batch run \
      --instance dgx1v.${MEMORY}g.${NGPU}.norm.beta \
      --name $NAME-$CONCEPT-ml-model.nomodel \
      --image "nvcr.io/nvidian/robotics/storm_kit:cuda_10.2_cudnn_driver418.40.04_20210701-154151" \
      --workspace concept_learning:concept_learning \
      --result /result \
      --port 6006 \
      --commandline  "
      export OMP_NUM_THREADS=1
      cd /concept_learning/concept_learning/src
      pip3 install shapely
      pip3 install h5py==2.10.0
      pip3 install moviepy
      python3 datasets/label_concept_data.py \
      --concept $CONCEPT --concept_model 'confrandgt_rawstate_'$TRAIN_AMT'_0.pt'
      "
  done
done