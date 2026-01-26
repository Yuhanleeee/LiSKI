import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re


def calibration_transweight(qwen253b, qwen253b_finetune, head_trans, trans_weight, site_layer, specific_layer=None):
    cali_head = None
    calibration_weight = {}

    for (name, param), (_, param_raw) in zip(qwen253b_finetune.named_parameters(), qwen253b.named_parameters()):
        layer_num = re.search(r'layers\.(\d+)\.', name)
        if layer_num == None:  ## embed
            if "embed" in name:
                calibration_weight[name] = param - param_raw - trans_weight[name]
            elif "lm_head" in name:
                if head_trans.shape[0] != qwen253b_finetune.lm_head.weight.shape[0]:
                    head_trans = torch.ones([qwen253b_finetune.lm_head.weight.shape[0], head_trans.shape[0]]) @ head_trans
                cali_head = qwen253b_finetune.lm_head.weight - qwen253b.lm_head.weight - head_trans
        else:
            layer_num = int(layer_num.group(1))
            # if layer_num in specific_layer:
            #     calibration_weight[name] = param - param_raw - trans_weight[name]
            ## last 24 layers
            if layer_num>=site_layer:
                calibration_weight[name] = param - param_raw - trans_weight[name]
            # if layer_num < 12:
            #     calibration_weight[name] = param - param_raw - trans_weight[name]
            # elif layer_num >= 24:
            #     calibration_weight[name] = param - param_raw - trans_weight[name.replace(str(layer_num), str(layer_num-12))]  
    if cali_head == None:
        if head_trans.shape[0] != qwen253b_finetune.lm_head.weight.shape[0]:
            head_trans = torch.ones([qwen253b_finetune.lm_head.weight.shape[0], head_trans.shape[0]]) @ head_trans
        cali_head = qwen253b_finetune.lm_head.weight - qwen253b.lm_head.weight - head_trans
    return cali_head, calibration_weight


def full_train_model_save(qwen253b, calibration_weight, calibration_weight_lmhead, weight_trans_qwen253b_lmhead, weight_trans_qwen253b, site_layer, specific_layer=None):
    ### save: raw_model+inject_knowledge+calibration_matrix  
    for name, param in qwen253b.named_parameters():
        if name in calibration_weight:
            layer_num = re.search(r'layers\.(\d+)\.', name)
            if layer_num == None:
                plust_weight = calibration_weight[name] + param + weight_trans_qwen253b[name]
            else:
                layer_num = int(layer_num.group(1))
                # if layer_num in specific_layer:
                #     plust_weight = calibration_weight[name] + param + weight_trans_qwen253b[name]
                if layer_num >= site_layer:
                    plust_weight = calibration_weight[name] + param + weight_trans_qwen253b[name]
                # if layer_num < 12:
                #     plust_weight = calibration_weight[name] + param + weight_trans_qwen253b[name]
                # elif layer_num >= 24:
                #     plust_weight= calibration_weight[name] + param + weight_trans_qwen253b[name.replace(str(layer_num), str(layer_num-12))]
            with torch.no_grad():
                param.copy_(plust_weight.to(param.device))
    with torch.no_grad():
        if weight_trans_qwen253b_lmhead.shape[0] != qwen253b.lm_head.weight.shape[0]:
            weight_trans_qwen253b_lmhead = torch.ones([qwen253b.lm_head.weight.shape[0], weight_trans_qwen253b_lmhead.shape[0]]) @ weight_trans_qwen253b_lmhead
        qwen253b.lm_head.weight.copy_((qwen253b.lm_head.weight + calibration_weight_lmhead + weight_trans_qwen253b_lmhead).to(qwen253b.lm_head.weight.device))
    return qwen253b
    

def calibration(qwen253b, qwen253b_finetune, svd_trans, svd_trans_head, specific_layer=None):
    cali_head = qwen253b_finetune.lm_head.weight - qwen253b.lm_head.weight - svd_trans_head[0] @ svd_trans_head[1].T
    calibration_weight = {}
    for (name, param), (_, param_raw) in zip(qwen253b_finetune.named_parameters(), qwen253b.named_parameters()):
        if len(param.shape) > 1:  ### remove bias no svd
            layer_num = re.search(r'layers\.(\d+)\.', name)
            if layer_num == None:  ## embed
                calibration_weight[name] = param - param_raw - svd_trans[name][0] @ svd_trans[name][1].T
            else:
                layer_num = int(layer_num.group(1))
                # if layer_num in specific_layer:
                #     calibration_weight[name] = param - param_raw - svd_trans[name][0] @ svd_trans[name][1].T
                if layer_num>=12:  ## last 24 layers
                    calibration_weight[name] = param - param_raw - svd_trans[name][0] @ svd_trans[name][1].T
                
                # if layer_num < 12:
                #     calibration_weight[name] = param - param_raw - svd_trans[name][0] @ svd_trans[name][1].T
                # elif layer_num >=24:
                #     calibration_weight[name] = param - param_raw - svd_trans[name.replace(str(layer_num), str(layer_num-12))][0] @ svd_trans[name.replace(str(layer_num), str(layer_num-12))][1].T
    return cali_head, calibration_weight


def model_save(qwen253b, calibration_weight, calibration_weight_lmhead):
    ### save: raw_model+calibration_matrix  
    for name, param in qwen253b.named_parameters():
        if name in calibration_weight:
            plust_weight = calibration_weight[name] + param
            with torch.no_grad():
                param.copy_(plust_weight.to(param.device))
    with torch.no_grad():
        qwen253b.lm_head.weight.copy_((qwen253b.lm_head.weight + calibration_weight_lmhead).to(qwen253b.lm_head.weight.device))
    return qwen253b



# qwen253b = AutoModelForCausalLM.from_pretrained('/data_1/zhl/ReverseDistillation/llms/Qwen2.5-3B-Instruct')
# qwen253b_valdata_finetune = AutoModelForCausalLM.from_pretrained('/data_1/zhl/ReverseDistillation/LLaMA-Factory/saves/qwen25_medcasereasoning/full/sft/medcasereasoning_reasoningdiagnostic_valdata_freezefirst12layer')
# trans_head = torch.load('/data_1/zhl/ReverseDistillation/weighttranslast24_delta_lmhead_medcasereasoning_val_reasoningdiagnostic.pth')
# trans_weight = torch.load('/data_1/zhl/ReverseDistillation/weighttranslast24_delta_medcasereasoning_val_reasoningdiagnostic.pth')
# calibration_weight_lmhead, calibration_weight = calibration_transweight(qwen253b, qwen253b_valdata_finetune, trans_head, trans_weight)
# trans_head_all = torch.load('/data_1/zhl/ReverseDistillation/weighttranslast24_delta_lmhead_medcasereasoning_all_reasoningdiagnostic.pth')
# trans_weight_all = torch.load('/data_1/zhl/ReverseDistillation/weighttranslast24_delta_medcasereasoning_all_reasoningdiagnostic.pth')
# qwen253b_cali_fulltrain = full_train_model_save(qwen253b, calibration_weight, calibration_weight_lmhead, trans_head_all, trans_weight_all)
# qwen253b_cali_fulltrain.save_pretrained('/data_1/zhl/ReverseDistillation/LLaMA-Factory/saves/qwen25_medcasereasoning/full/calibration/medcasereasoning_reasoningdiagnostic_calibration_weighttrans_last24layer_fullsft')
# quit()
# quit()

# qwen253b = AutoModelForCausalLM.from_pretrained('/data_1/zhl/ReverseDistillation/llms/Qwen2.5-3B-Instruct')
# qwen253b_valdata_finetune = AutoModelForCausalLM.from_pretrained('/data_1/zhl/ReverseDistillation/LLaMA-Factory/saves/qwen25_medcasereasoning/full/sft/medcasereasoning_val_reasoningdiagnostic_253b_specificlayer_8epoch')
# trans_head = torch.load('/data_1/zhl/ReverseDistillation/weighttransspecific_layer_delta_lmhead_medcasereasoning_val_reasoningdiagnostic.pth')
# trans_weight = torch.load('/data_1/zhl/ReverseDistillation/weighttransspecific_layer_delta_medcasereasoning_val_reasoningdiagnostic.pth')
# specific_layer = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35]
# calibration_weight_lmhead, calibration_weight = calibration_transweight(qwen253b, qwen253b_valdata_finetune, trans_head, trans_weight, specific_layer)

# trans_head_all = torch.load('/data_1/zhl/ReverseDistillation/weighttransspecific_delta_lmhead_medcasereasoning_all_reasoningdiagnostic.pth')
# trans_weight_all = torch.load('/data_1/zhl/ReverseDistillation/weighttransspecific_delta_medcasereasoning_all_reasoningdiagnostic.pth')
# qwen253b_cali_fulltrain = full_train_model_save(qwen253b, calibration_weight, calibration_weight_lmhead, trans_head_all, trans_weight_all, specific_layer)
# qwen253b_cali_fulltrain.save_pretrained('/data_1/zhl/ReverseDistillation/LLaMA-Factory/saves/qwen25_medcasereasoning/full/calibration/medcasereasoning_reasoningdiagnostic_calibration_weighttrans_specificlayer_fullsft')
# quit()

# qwen253b = AutoModelForCausalLM.from_pretrained('/data_1/zhl/ReverseDistillation/llms/Qwen2.5-3B-Instruct')
# qwen253b_valdata_finetune = AutoModelForCausalLM.from_pretrained('/data_1/zhl/ReverseDistillation/LLaMA-Factory/saves/qwen25_medcasereasoning/full/sft/medcasereasoning_val_reasoningdiagnostic_253b_specificlayer_8epoch')
# svd_trans_head = torch.load('/data_1/zhl/ReverseDistillation/svd_delta_weighttransspecific_layer_lmhead_medcasereasoning_val_reasoningdiagnostic.pth')
# svd_trans = torch.load('/data_1/zhl/ReverseDistillation/svd_delta_weighttransspecific_layer_medcasereasoning_val_reasoningdiagnostic.pth')
# specific_layer = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35]
# calibration_weight_lmhead, calibration_weight = calibration(qwen253b, qwen253b_valdata_finetune, svd_trans, svd_trans_head, specific_layer)
# qwen253b_cali = model_save(qwen253b, calibration_weight, calibration_weight_lmhead)
# qwen253b_cali.save_pretrained('/data_1/zhl/ReverseDistillation/LLaMA-Factory/saves/qwen25_medcasereasoning/full/calibration/medcasereasoning_reasoningdiagnostic_epoch8_3B_valcalibration_specificlayer')



## calibration + kj + raw
# path_llm = '/data/yakun_data/A-mem/models/Qwen2.5-7B-Instruct/'
# path_llm = '/data/yakun_data/A-mem/models/phi-2'
# path_llm = '/data/yakun_data/A-mem/models/Llama-3.2-3B-Instruct/'
path_llm = '/data/yakun_data/A-mem/models/Qwen2.5-3B-Instruct/'

# path_llm_valfinetunefreeze = '/root/data/kj/LLaMA-Factory/saves/medcasereasoning/full/sft/qwen257B_medcasereasoning_val_epoch6_freeze_firstlayers'
# path_trans_head_val = 'kj_weight/medcasereasoning/valdata_kj_lastlayers_lmhead.pth'
# path_trans_weight_val = '/root/data/kj/kj_weight/medcasereasoning/valdata_kj_lastlayers.pth'
# path_trans_weight_all = "/root/data/kj/kj_weight/medcasereasoning/alldata_kj_lastlayers.pth"
# path_trans_head_all = "/root/data/kj/kj_weight/medcasereasoning/alldata_kj_lastlayers_lmhead.pth"

# path_llm_valfinetunefreeze = '/data/yakun_data/kj/LLaMA-Factory/saves/medcasereasoning/full/sft/qwen257B_medcasereasoning_val_epoch6_allsft/checkpoint-252'
# path_trans_head_val = "/data/yakun_data/kj/kj_weight/medcasereasoning/qwen253B27B_valdata_kj_lastlayers_lmhead.pth"
# path_trans_weight_val = "/data/yakun_data/kj/kj_weight/medcasereasoning/qwen253B27B_valdata_kj_lastlayers.pth"
# path_trans_weight_all = "/data/yakun_data/kj/kj_weight/medcasereasoning/qwen253B27B_alldata_kj_lastlayers.pth"
# path_trans_head_all = "/data/yakun_data/kj/kj_weight/medcasereasoning/qwen253B27B_alldata_kj_lastlayers_lmhead.pth"

# path_llm_valfinetunefreeze = '/data/yakun_data/kj/LLaMA-Factory/saves/CHEBI20/full/sft/qwen257b_val_6epoch_freezefirst4layers'
# path_trans_head_val = '/data/yakun_data/kj/kj_weight/chebi20/qwen2505B27B_valdata_kj_lastlayers_lmhead.pth'
# path_trans_weight_val = '/data/yakun_data/kj/kj_weight/chebi20/qwen2505B27B_valdata_kj_lastlayers.pth'
# path_trans_weight_all = '/data/yakun_data/kj/kj_weight/chebi20/qwen2505B27B_alldata_kj_lastlayers.pth'
# path_trans_head_all = '/data/yakun_data/kj/kj_weight/chebi20/qwen2505B27B_alldata_kj_lastlayers_lmhead.pth'

# path_llm_valfinetunefreeze = '/data/yakun_data/kj/LLaMA-Factory/saves/CHEBI20/full/sft/qwen257b_val_6epoch_sft'
# path_trans_head_val = '/data/yakun_data/kj/kj_weight/chebi20/qwen253B27B_valdata_kj_lastlayers_lmhead.pth'
# path_trans_weight_val = '/data/yakun_data/kj/kj_weight/chebi20/qwen253B27B_valdata_kj_lastlayers.pth'
# path_trans_weight_all = '/data/yakun_data/kj/kj_weight/chebi20/qwen253B27B_alldata_kj_lastlayers.pth'
# path_trans_head_all = '/data/yakun_data/kj/kj_weight/chebi20/qwen253B27B_alldata_kj_lastlayers_lmhead.pth'


# path_llm_valfinetunefreeze = '/data/yakun_data/kj/LLaMA-Factory/saves/CHEBI20/full/sft/phi2_val_6epoch_freeze_first8layers'
# path_trans_weight_val = '/data/yakun_data/kj/kj_weight/chebi20/phi/phi1522_valdata_kj_lastlayers.pth'
# path_trans_head_val = '/data/yakun_data/kj/kj_weight/chebi20/phi/phi1522_valdata_kj_lastlayers_lmhead.pth'
# path_trans_weight_all = '/data/yakun_data/kj/kj_weight/chebi20/phi/phi1522_alldata_kj_lastlayers.pth'
# path_trans_head_all = '/data/yakun_data/kj/kj_weight/chebi20/phi/phi1522_alldata_kj_lastlayers_lmhead.pth'

# path_llm_valfinetunefreeze = '/data/yakun_data/kj/LLaMA-Factory/saves/CHEBI20/full/sft/llama323B_val_6epoch_freezefirst12layers'
# path_trans_weight_val = '/data/yakun_data/kj/kj_weight/chebi20/llama/llama321B23B_valdata_kj_lastlayers.pth'
# path_trans_head_val = '/data/yakun_data/kj/kj_weight/chebi20/llama/llama321B23B_valdata_kj_lastlayers_lmhead.pth'
# path_trans_weight_all = '/data/yakun_data/kj/kj_weight/chebi20/llama/llama321B23B_alldata_kj_lastlayers.pth'
# path_trans_head_all = '/data/yakun_data/kj/kj_weight/chebi20/llama/llama321B23B_alldata_kj_lastlayers_lmhead.pth'

# path_trans_weight_val = '/data/yakun_data/kj/kj_weight/chebi20/qwen/qwen2505B23B_005sample_valdata_kj_lastlayers.pth'
# path_trans_head_val = '/data/yakun_data/kj/kj_weight/chebi20/qwen/qwen2505B23B_005sample_valdata_kj_lastlayers_lmhead.pth'
# path_llm_valfinetunefreeze = '/data/yakun_data/kj/LLaMA-Factory/saves/CHEBI20/full/sft/qwen253B_sft_6epoch_CHEBI20_005sample_freezelast12layers'
# path_trans_weight_all = '/data/yakun_data/kj/kj_weight/chebi20/qwen/qwen2505B23B_alldata_kj_lastlayers.pth'
# path_trans_head_all = '/data/yakun_data/kj/kj_weight/chebi20/qwen/qwen2505B23B_alldata_kj_lastlayers_lmhead.pth'

path_trans_weight_val = '/data/yakun_data/kj/kj_weight/medcasereasoning/qwen/qwen2505B23B_01sample_valdata_kj_lastlayers.pth'
path_trans_head_val = '/data/yakun_data/kj/kj_weight/medcasereasoning/qwen/qwen2505B23B_01sample_valdata_kj_lastlayers_lmhead.pth'
path_llm_valfinetunefreeze = '/data/yakun_data/kj/LLaMA-Factory/saves/medcasereasoning/full/sft/qwen253B_sft_6epoch_medcasereasoning_reasoningdiagnostic_01sample_freezelast12layers'

path_trans_weight_all = '/data/yakun_data/kj/kj_weight/medcasereasoning/qwen/qwen2505B23B_alldata_kj_lastlayers.pth'
path_trans_head_all = '/data/yakun_data/kj/kj_weight/medcasereasoning/qwen/qwen2505B23B_alldata_kj_lastlayers_lmhead.pth'

llm = AutoModelForCausalLM.from_pretrained(path_llm)
llm_valdata_finetune = AutoModelForCausalLM.from_pretrained(path_llm_valfinetunefreeze)
trans_head_val = torch.load(path_trans_head_val)
trans_weight_val = torch.load(path_trans_weight_val)
calibration_weight_lmhead, calibration_weight = calibration_transweight(llm, llm_valdata_finetune, trans_head_val, trans_weight_val, site_layer=12)  ## site_layer=0: qwen253Bto7B, site_layer=4: qwen2505Bto7B, site_layer=12, llama3.21Bto3B
trans_head_all = torch.load(path_trans_head_all)
trans_weight_all = torch.load(path_trans_weight_all)
qwen253b_cali_fulltrain = full_train_model_save(llm, calibration_weight, calibration_weight_lmhead, trans_head_all, trans_weight_all, site_layer=12)
# qwen253b_cali_fulltrain.save_pretrained('kj_weight/chebi20/llama/chebi20_calibration_kjinject_lastlayers_llama321Bto3B')
qwen253b_cali_fulltrain.save_pretrained('kj_weight/medcasereasoning/medcasereasoning_calibration_kjinject_lastlayers_qwen2505Bto3B_01sample')

quit()
quit()
quit()





qwen253b = AutoModelForCausalLM.from_pretrained('/data_1/zhl/ReverseDistillation/llms/Qwen2.5-3B-Instruct')
# qwen253b_valdata_finetune = AutoModelForCausalLM.from_pretrained('/data_1/zhl/ReverseDistillation/LLaMA-Factory/saves/qwen25_medcasereasoning/full/sft/medcasereasoning_reasoningdiagnostic_valdata_freezefirst12layer')
qwen253b_valdata_finetune = AutoModelForCausalLM.from_pretrained('/data_1/zhl/ReverseDistillation/LLaMA-Factory/saves/qwen25_medcasereasoning/full/sft/medcasereasoning_reasoningdiagnostic_valdata_epoch6_3B_last24layeremblmhead')

svd_trans_head = torch.load('/data_1/zhl/ReverseDistillation/kj_weight/svd_medcasereasoning_valdata_kjnoT_last24layer_lmhead_reasoningdiagnostic.pth')
svd_trans = torch.load('/data_1/zhl/ReverseDistillation/kj_weight/svd_medcasereasoning_valdata_kjnoT_last24layer_reasoningdiagnostic.pth')
calibration_weight_lmhead, calibration_weight = calibration(qwen253b, qwen253b_valdata_finetune, svd_trans, svd_trans_head, specific_layer=None)
qwen253b_cali = model_save(qwen253b, calibration_weight, calibration_weight_lmhead)
qwen253b_cali.save_pretrained('/data_1/zhl/ReverseDistillation/kj_weight/calibration/valdata_kjnoT_last24layer_reasoningdiagnostic')







