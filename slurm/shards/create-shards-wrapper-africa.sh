#!/bin/bash
# launcher for train data shard creation (2-year blocks, debug-safe)

JOB_SCRIPT="/home/users/mendrika/Object-Based-LSTMConv/slurm/shards/create-shards-africa.sh"

LEAD_TIMES=("030" "060" "090" "120")

YEAR_BLOCKS=(
  "2004 2005"
  "2006 2007"
  "2008 2009"
  "2010 2011"
  "2012 2013"
  "2014 2015"
  "2016 2017"
  "2018 2019"
  "2020 2021"
  "2022 2022"
)

PARTITION="train"

for LEAD_TIME in "${LEAD_TIMES[@]}"; do
    for Y in "${YEAR_BLOCKS[@]}"; do
        set -- $Y
        YEAR_START=$1
        YEAR_END=$2

        echo "Submitting train job | lead=${LEAD_TIME} | years=${YEAR_START}-${YEAR_END}"

        sbatch \
            -J "train${LEAD_TIME}_${YEAR_START}_${YEAR_END}" \
            "$JOB_SCRIPT" \
            "$PARTITION" \
            "$LEAD_TIME" \
            "$YEAR_START" \
            "$YEAR_END"

        sleep 1
    done
done

echo "All train shard jobs submitted successfully."
