import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re


def calibration_transweight(baselm, baselm_finetune, head_trans, trans_weight, site_layer, specific_layer=None, lambda_kj=1):
    cali_head = None
    calibration_weight = {}

    for (name, param), (_, param_raw) in zip(baselm_finetune.named_parameters(), baselm.named_parameters()):
        layer_num = re.search(r'layers\.(\d+)\.', name)
        if layer_num == None:  ## embed
            if "embed" in name:
                calibration_weight[name] = param - param_raw - lambda_kj*trans_weight[name]
            elif "lm_head" in name:
                if head_trans.shape[0] != baselm_finetune.lm_head.weight.shape[0]:
                    # head_trans = torch.eye([baselm_finetune.lm_head.weight.shape[0], head_trans.shape[0]]) @ head_trans
                    head_trans = torch.ones([baselm_finetune.lm_head.weight.shape[0], head_trans.shape[0]]) @ head_trans
                cali_head = baselm_finetune.lm_head.weight - baselm.lm_head.weight - lambda_kj*head_trans
        else:
            layer_num = int(layer_num.group(1))
            ## specific layers
            # if layer_num in specific_layer:
            #     calibration_weight[name] = param - param_raw - lambda_kj*trans_weight[name]
            ## last layers
            if layer_num>=site_layer:
                calibration_weight[name] = param - param_raw - lambda_kj*trans_weight[name]
            # if layer_num < 12:
            #     calibration_weight[name] = param - param_raw - lambda_kj*trans_weight[name]
            # elif layer_num >= 24:
            #     calibration_weight[name] = param - param_raw -lambda_kj* trans_weight[name.replace(str(layer_num), str(layer_num-12))]  
    if cali_head == None:
        if head_trans.shape[0] != baselm_finetune.lm_head.weight.shape[0]:
            # head_trans = torch.ones([baselm_finetune.lm_head.weight.shape[0], head_trans.shape[0]]) @ head_trans
            head_trans = torch.eye([baselm_finetune.lm_head.weight.shape[0], head_trans.shape[0]]) @ head_trans
        cali_head = baselm_finetune.lm_head.weight - baselm.lm_head.weight - lambda_kj*head_trans
    return cali_head, calibration_weight

def full_train_model_save_iterative(baselm, calibration_weight, calibration_weight_lmhead, weight_trans_baselm_lmhead, weight_trans_baselm, site_layer, specific_layer=None, lambda_kj=1, xi=0.0001):
    ### save with an iterative manner: raw_model+inject_knowledge+calibration_matrix
    for name, param in baselm.named_parameters():
        if name in calibration_weight:
            layer_num = re.search(r'layers\.(\d+)\.', name)
            if layer_num == None:
                plust_weight = lambda_kj*calibration_weight[name]*(1+0.0001*torch.randn_like(calibration_weight[name])) + param + lambda_kj*weight_trans_baselm[name]
            else:
                layer_num = int(layer_num.group(1))
                ## specific layers
                # if layer_num in specific_layer:
                #     plust_weight = lambda_kj*calibration_weight[name]*(1+xi*torch.randn_like(calibration_weight[name])) + param + lambda_kj*weight_trans_baselm[name]
                if layer_num >= site_layer:
                    plust_weight = lambda_kj*calibration_weight[name]*(1+xi*torch.randn_like(calibration_weight[name])) + param + lambda_kj*weight_trans_baselm[name]
                # if layer_num < 12:
                #     plust_weight = lambda_kj*calibration_weight[name]*(1+xi*torch.randn_like(calibration_weight[name])) + param + lambda_kj*weight_trans_baselm[name]
                # elif layer_num >= 24:
                #     plust_weight= lambda_kj*calibration_weight[name]*(1+xi*torch.randn_like(calibration_weight[name])) + param + lambda_kj*weight_trans_baselm[name.replace(str(layer_num), str(layer_num-12))]
            with torch.no_grad():
                param.copy_(plust_weight.to(param.device))
    with torch.no_grad():
        if weight_trans_baselm_lmhead.shape[0] != baselm.lm_head.weight.shape[0]:
            # weight_trans_baselm_lmhead = torch.eye([baselm.lm_head.weight.shape[0], weight_trans_baselm_lmhead.shape[0]]) @ weight_trans_baselm_lmhead
            weight_trans_baselm_lmhead = torch.ones([baselm.lm_head.weight.shape[0], weight_trans_baselm_lmhead.shape[0]]) @ weight_trans_baselm_lmhead
        baselm.lm_head.weight.copy_((baselm.lm_head.weight + lambda_kj*calibration_weight_lmhead*(1+xi*torch.randn_like(calibration_weight_lmhead)) + lambda_kj*weight_trans_baselm_lmhead).to(baselm.lm_head.weight.device))
    return baselm

def full_train_model_save(baselm, calibration_weight, calibration_weight_lmhead, weight_trans_baselm_lmhead, weight_trans_baselm, site_layer, specific_layer=None, lambda_kj=1):
    ### save: raw_model+inject_knowledge+calibration_matrix  
    for name, param in baselm.named_parameters():
        if name in calibration_weight:
            layer_num = re.search(r'layers\.(\d+)\.', name)
            if layer_num == None:
                plust_weight = lambda_kj*calibration_weight[name] + param + lambda_kj*weight_trans_baselm[name]
            else:
                layer_num = int(layer_num.group(1))
                ## specific layers
                # if layer_num in specific_layer:
                #     plust_weight = lambda_kj*calibration_weight[name] + param + lambda_kj*weight_trans_baselm[name]
                if layer_num >= site_layer:
                    plust_weight = lambda_kj*calibration_weight[name] + param + lambda_kj*weight_trans_baselm[name]
                # if layer_num < 12:
                #     plust_weight = lambda_kj*calibration_weight[name] + param + lambda_kj*weight_trans_baselm[name]
                # elif layer_num >= 24:
                #     plust_weight= lambda_kj*calibration_weight[name] + param + lambda_kj*weight_trans_baselm[name.replace(str(layer_num), str(layer_num-12))]
            with torch.no_grad():
                param.copy_(plust_weight.to(param.device))
    with torch.no_grad():
        if weight_trans_baselm_lmhead.shape[0] != baselm.lm_head.weight.shape[0]:
            # weight_trans_baselm_lmhead = torch.eye([baselm.lm_head.weight.shape[0], weight_trans_baselm_lmhead.shape[0]]) @ weight_trans_baselm_lmhead
            weight_trans_baselm_lmhead = torch.ones([baselm.lm_head.weight.shape[0], weight_trans_baselm_lmhead.shape[0]]) @ weight_trans_baselm_lmhead
        baselm.lm_head.weight.copy_((baselm.lm_head.weight + lambda_kj*calibration_weight_lmhead + lambda_kj*weight_trans_baselm_lmhead).to(baselm.lm_head.weight.device))
    return baselm
    

def calibration_withsvd(baselm, baselm_finetune, svd_trans, svd_trans_head, specific_layer=None, layer_site=12, lambda_kj=1):
    cali_head = baselm_finetune.lm_head.weight - baselm.lm_head.weight - lambda_kj*(svd_trans_head[0] @ svd_trans_head[1].T)
    calibration_weight = {}
    for (name, param), (_, param_raw) in zip(baselm_finetune.named_parameters(), baselm.named_parameters()):
        if len(param.shape) > 1:  ### remove bias no svd
            layer_num = re.search(r'layers\.(\d+)\.', name)
            if layer_num == None:  ## embed
                calibration_weight[name] = param - param_raw - lambda_kj*(svd_trans[name][0] @ svd_trans[name][1].T)
            else:
                layer_num = int(layer_num.group(1))
                ## specific layers
                # if layer_num in specific_layer:
                #     calibration_weight[name] = param - param_raw - lambda_kj*(svd_trans[name][0] @ svd_trans[name][1].T)
                if layer_num>=layer_site:  ## last 24 layers
                    calibration_weight[name] = param - param_raw - lambda_kj*(svd_trans[name][0] @ svd_trans[name][1].T)
                
                # if layer_num < 12:
                #     calibration_weight[name] = param - param_raw - lambda_kj*(svd_trans[name][0] @ svd_trans[name][1].T)
                # elif layer_num >=24:
                #     calibration_weight[name] = param - param_raw - lambda_kj*(svd_trans[name.replace(str(layer_num), str(layer_num-12))][0] @ svd_trans[name.replace(str(layer_num), str(layer_num-12))][1].T)
    return cali_head, calibration_weight


def model_save(baselm, calibration_weight, calibration_weight_lmhead, lambda_kj=1):
    ### save: raw_model+calibration_matrix  
    for name, param in baselm.named_parameters():
        if name in calibration_weight:
            plust_weight = lambda_kj*calibration_weight[name] + param
            with torch.no_grad():
                param.copy_(plust_weight.to(param.device))
    with torch.no_grad():
        baselm.lm_head.weight.copy_((baselm.lm_head.weight + lambda_kj*calibration_weight_lmhead).to(baselm.lm_head.weight.device))
    return baselm


## calibration + kj + raw
path_llm = 'models/Qwen2.5-3B-Instruct/'


path_trans_weight_val = 'kj_weight_calibration/.pth'
path_trans_head_val = 'kj_weight_lmhead_calibration/.pth'
path_llm_valfinetune = ''  ## llm finetuned with calibration data

path_trans_weight_all = 'kj_weight_all/.pth'
path_trans_head_all = 'kj_weight_lmhead_all/.pth'

llm = AutoModelForCausalLM.from_pretrained(path_llm)
llm_valdata_finetune = AutoModelForCausalLM.from_pretrained(path_llm_valfinetune)
trans_head_val = torch.load(path_trans_head_val)
trans_weight_val = torch.load(path_trans_weight_val)
calibration_weight_lmhead, calibration_weight = calibration_transweight(llm, llm_valdata_finetune, trans_head_val, trans_weight_val, site_layer=12)  ## site_layer=0: baselmto7B, site_layer=4: qwen2505Bto7B, site_layer=12, llama3.21Bto3B
trans_head_all = torch.load(path_trans_head_all)
trans_weight_all = torch.load(path_trans_weight_all)
baselm_cali_fulltrain = full_train_model_save(llm, calibration_weight, calibration_weight_lmhead, trans_head_all, trans_weight_all, site_layer=12)
baselm_cali_fulltrain.save_pretrained('')

"""
baselm = AutoModelForCausalLM.from_pretrained('models/Qwen2.5-3B-Instruct')
baselm_valdata_finetune = AutoModelForCausalLM.from_pretrained(path_llm_valfinetune)

svd_trans_head = torch.load('kj_weight_lmhead_svd/.pth')
svd_trans = torch.load('kj_weight_svd/.pth')
calibration_weight_lmhead, calibration_weight = calibration_withsvd(baselm, baselm_valdata_finetune, svd_trans, svd_trans_head, specific_layer=None, layer_site=12)
baselm_cali = model_save(baselm, calibration_weight, calibration_weight_lmhead)
baselm_cali.save_pretrained('')
"""






