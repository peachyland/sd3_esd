

# MY_CMD="python esd_diffusers.py --erase_concept 'Van Gogh' --train_method 'xattn'"

# echo $MY_CMD
# echo ${MY_CMD}>>local_history.log
CUDA_VISIBLE_DEVICES='3' python esd_diffusers.py --erase_concept 'Van Gogh' --train_method 'xattn'

# CUDA_VISIBLE_DEVICES='2' HF_HOME="/egr/research-dselab/renjie3/.cache" python sd3.py
