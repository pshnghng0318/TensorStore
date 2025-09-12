#!/bin/bash
LOG="slice_result_1T.log"
PNG="slice_result_1T.png"

psrecord "./bazel-bin/examples/extract_slice \
  --output_file=/home/pshnghng/Softwares/slice/tensorstore/slice_ncomp.png \
  --input_spec='{
    \"driver\": \"zarr\",
    \"kvstore\": {
      \"driver\": \"file\",
      \"path\": \"/mnt/ACDC_1TB/askap_hydra_extragalactic_256_NotCompressed_v2.zarr/SKY/\"
    }
  }'" \
  --interval 0.01 --include-children --log "$LOG" --plot "$PNG"
