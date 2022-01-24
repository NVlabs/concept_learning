WSNAME=concept_learning
MOUNTDIR=/home/abobu/Project/ngc_ws/${WSNAME}

# # mount workspace
# ngc workspace unmount ${MOUNTDIR}
# ngc workspace mount ${WSNAME} ${MOUNTDIR} --mode RW

# sync files
SCRIPTDIR=$(dirname $(readlink -f "$0"))
PROJECTNAME=$(basename "$(dirname "$SCRIPTDIR")")
echo ${SCRIPTDIR}
echo ${MOUNTDIR}/${PROJECTNAME}

rsync -rlvczP \
  --exclude '__pycache__' \
  --exclude '.git' \
  --exclude 'checkpoints' \
  --exclude '*.pyc' \
  --exclude 'graspnet.egg-info' \
  --exclude '.eggs' \
  --exclude 'data/concept_*/' \
  --exclude 'data/g_*/' \
  --exclude 'data/test_shapenet/' \
  --exclude 'scripts/' \
  --exclude 'models*/' \
  --exclude 'results/' \
  ${SCRIPTDIR}/.. ${MOUNTDIR}/${PROJECTNAME}/
