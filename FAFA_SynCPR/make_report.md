# Make report 10 sample
 
 ```bash
cd FAFA_SynCPR/src

python inference_visual.py \
    --checkpoint ../output/cpr/FAFA_experiment/saved_models/tuned_recall_at1_step.pt \
    --itcpr-root ../../ITCPR \
    --output-dir ../../result
```