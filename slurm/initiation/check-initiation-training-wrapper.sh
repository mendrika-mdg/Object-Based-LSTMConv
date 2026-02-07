#!/bin/bash
# launcher for CI–core association jobs

JOB_SCRIPT="/home/users/mendrika/Object-Based-LSTMConv/slurm/initiation/check-initiation-training.sh"

# lead times in minutes
LEAD_TIMES=("030" "060" "090" "120")

# years and months to process
YEARS=("2004" "2005" "2006" "2007" "2008" "2009" "2010" "2011" "2012" "2013" "2014" "2015" "2016" "2017" "2018" "2019" "2020" "2021" "2022" "2023")
MONTHS=("01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12")

for LEAD_TIME in "${LEAD_TIMES[@]}"; do
    for YEAR in "${YEARS[@]}"; do
        for MONTH in "${MONTHS[@]}"; do
            echo "Submitting job for lead_time=${LEAD_TIME}, year=${YEAR}, month=${MONTH}..."
            sbatch -J "t${LEAD_TIME}_${YEAR}${MONTH}" \
                "$JOB_SCRIPT" "$LEAD_TIME" "$YEAR" "$MONTH"
            sleep 2
        done
    done
done

echo "All jobs submitted successfully."
