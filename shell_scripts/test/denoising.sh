net=$1
testdata=$2
norm=$3
mfolder=$4

if [ "$net" = "ANN" ]; then
  shape="(32, 32, 32)"
else
  shape="(32, 32, 8)"
fi
python src/denoising_test.py \
    --net ${net} \
    --folder ${testdata} \
    --mfolder ${mfolder} \
    --testtype 1 \
    --shape "${shape}" \
     --no_to_snr 3 \
    --norm ${norm}
