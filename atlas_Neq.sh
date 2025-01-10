#!/bin/bash

# Create a directory to store the analysis files
mkdir -p analysis_data

# Download, unzip, and process RMSF files
for name in $(cat atlas.csv | grep -v name | awk -F ',' '{print $1}'); do
    # Download the analysis zip file
    wget https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/${name}/${name}_analysis.zip -P analysis_data

    # Unzip the analysis file
    unzip analysis_data/${name}_analysis.zip -d analysis_data/${name}
    
    # Remove the zip file
    rm analysis_data/${name}_analysis.zip

    # Move the RMSF.tsv file to the main analysis_data directory
    if [ -f analysis_data/${name}/${name}_Neq.tsv ]; then
        mv analysis_data/${name}/${name}_Neq.tsv analysis_data/
    fi

    # Remove the remaining extracted files and directories
    rm -r analysis_data/${name}
done
