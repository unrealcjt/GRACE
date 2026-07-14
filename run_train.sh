# !bin/bash
# train from begin
python train.py --dataset="iemocap" --modalities="atv" --from_begin --epochs=50 --learning_rate=0.0002 --optimizer="adamw" --drop_rate=0.55 --batch_size=10 --rnn="transformer" --use_speaker  --edge_type="temp_multi" --wp=11 --wf=9  --gcn_conv="rgcn" --use_graph_transformer --graph_transformer_nheads=7  --use_crossmodal --num_crossmodal=3 --num_self_att=4 --crossmodal_nheads=2 --self_att_nheads=2

