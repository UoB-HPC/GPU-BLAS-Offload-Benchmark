#!/bin/bash
numactl -N3 -m3 ./gpu-blob_a1b0 -i 100 -o CSV_a1b0_1
numactl -N3 -m3 ./gpu-blob_a1b0 -i 100 -o CSV_a1b0_2
numactl -N3 -m3 ./gpu-blob_a1b0 -i 100 -o CSV_a1b0_3


numactl -N3 -m3 ./gpu-blob_a4b0 -i 100 -o CSV_a4b0_1
numactl -N3 -m3 ./gpu-blob_a4b0 -i 100 -o CSV_a4b0_2
numactl -N3 -m3 ./gpu-blob_a4b0 -i 100 -o CSV_a4b0_3


numactl -N3 -m3 ./gpu-blob_a1b2 -i 100 -o CSV_a1b2_1
numactl -N3 -m3 ./gpu-blob_a1b2 -i 100 -o CSV_a1b2_2
numactl -N3 -m3 ./gpu-blob_a1b2 -i 100 -o CSV_a1b2_3
