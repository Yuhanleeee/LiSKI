import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re


def trans_weight_lastlayers_llm(model_05_param_name, model_3_param_name, layer_site):
    """
    layers of slm over than that of llm, e.g., qwen253b to qwen257b
    """
    trans_weight = {}
    for name in model_3_param_name:
        if 'lm_head' not in name:
            if 'layers' in name:
                layer_num = int(re.search(r'layers\.(\d+)\.', name).group(1))
                name_trans = name.replace(str(layer_num), str(layer_num+layer_site))
                model_05_tempweight = model_05_param_name[name_trans]
                model_3_tempweight = model_3_param_name[name]
            else:
                model_05_tempweight = model_05_param_name[name]
                model_3_tempweight = model_3_param_name[name]
            if len(model_3_param_name[name].shape) == 1:
                # temp = (model_05_tempweight.unsqueeze(0).T) @ model_05_tempweight.unsqueeze(0)
                # w = torch.linalg.pinv(temp) @ (model_05_tempweight.unsqueeze(0).T @ model_3_tempweight.unsqueeze(0))
                w = torch.linalg.pinv(model_05_tempweight.unsqueeze(0)) @ model_3_tempweight.unsqueeze(0)
            else:
                if model_3_tempweight.shape[0] > model_3_tempweight.shape[1]:  ## right_w
                    # onesw = torch.ones([model_05_tempweight.shape[0], model_3_tempweight.shape[0]])
                    # w = torch.linalg.pinv(model_05_tempweight.T @ model_05_tempweight) @ model_05_tempweight.T @ torch.linalg.pinv(onesw.T @ onesw) @ onesw.T @ model_3_tempweight
                    # iw = torch.eye(model_3_tempweight.shape[0], model_05_tempweight.shape[0])
                    # w = torch.linalg.pinv(model_05_tempweight.T @ model_05_tempweight) @ model_05_tempweight.T @ iw.T @ model_3_tempweight
                    ## noT, best
                    B = torch.ones([model_3_tempweight.shape[0], model_05_tempweight.shape[0]]) @ model_05_tempweight
                    w = torch.linalg.pinv(B) @ model_3_tempweight  ## right_w
                else:  ## left_w
                    # onesw = torch.ones([model_05_tempweight.shape[1], model_3_tempweight.shape[1]])
                    # w = model_3_tempweight @ onesw.T @ torch.linalg.pinv(onesw @ onesw.T) @ model_05_tempweight.T @ torch.linalg.pinv(model_05_tempweight @ model_05_tempweight.T)
                    # iw = torch.eye(model_05_tempweight.shape[1], model_3_tempweight.shape[1])
                    # w = model_3_tempweight @ iw.T @ model_05_tempweight.T @ torch.linalg.pinv(model_05_tempweight @ model_05_tempweight.T)
                    ## noT, best
                    B = model_05_tempweight @ torch.ones([model_05_tempweight.shape[1], model_3_tempweight.shape[1]])
                    w = model_3_tempweight @ torch.linalg.pinv(B)  ## left_w
            trans_weight[name] = w
    return trans_weight


def trans_weight_lastlayers(model_05_param_name, model_3_param_name, layer_site):
    """
    layers of slm less than that of llm, e.g., qwen2515b to qwen253b
    """
    trans_weight = {}
    for name in model_05_param_name:
        if 'layers' in name:
            layer_num = int(re.search(r'layers\.(\d+)\.', name).group(1))
            model_05_tempweight = model_05_param_name[name]
            name_trans = name.replace(str(layer_num), str(layer_num+layer_site), 1)
            model_3_tempweight = model_3_param_name[name_trans]
        else:
            name_trans = name 
            model_05_tempweight = model_05_param_name[name]
            model_3_tempweight = model_3_param_name[name]
        if len(model_3_param_name[name_trans].shape) == 1:
            # temp = (model_05_tempweight.unsqueeze(0).T) @ model_05_tempweight.unsqueeze(0)
            # w = torch.linalg.pinv(temp) @ (model_05_tempweight.unsqueeze(0).T @ model_3_tempweight.unsqueeze(0))
            w = torch.linalg.pinv(model_05_tempweight.unsqueeze(0)) @ model_3_tempweight.unsqueeze(0)
        else:
            if model_3_tempweight.shape[0] > model_3_tempweight.shape[1]:  ## right_w
                # onesw = torch.ones([model_05_tempweight.shape[0], model_3_tempweight.shape[0]])
                # w = torch.linalg.pinv(model_05_tempweight.T @ model_05_tempweight) @ model_05_tempweight.T @ torch.linalg.pinv(onesw.T @ onesw) @ onesw.T @ model_3_tempweight
                # iw = torch.eye(model_3_tempweight.shape[0], model_05_tempweight.shape[0])
                # w = torch.linalg.pinv(model_05_tempweight.T @ model_05_tempweight) @ model_05_tempweight.T @ iw.T @ model_3_tempweight
                ## noT, best
                B = torch.ones([model_3_tempweight.shape[0], model_05_tempweight.shape[0]]) @ model_05_tempweight
                w = torch.linalg.pinv(B) @ model_3_tempweight  ## right_w
            else:  ## left_w
                # onesw = torch.ones([model_05_tempweight.shape[1], model_3_tempweight.shape[1]])
                # w = model_3_tempweight @ onesw.T @ torch.linalg.pinv(onesw @ onesw.T) @ model_05_tempweight.T @ torch.linalg.pinv(model_05_tempweight @ model_05_tempweight.T)
                # iw = torch.eye(model_05_tempweight.shape[1], model_3_tempweight.shape[1])
                # w = model_3_tempweight @ iw.T @ model_05_tempweight.T @ torch.linalg.pinv(model_05_tempweight @ model_05_tempweight.T)
                ## noT, best
                B = model_05_tempweight @ torch.ones([model_05_tempweight.shape[1], model_3_tempweight.shape[1]])
                w = model_3_tempweight @ torch.linalg.pinv(B)  ## left_w
        trans_weight[name_trans] = w
    return trans_weight


def trans_weight_gradscore_layer(model_05_param_name, model_3_param_name, specific_layer):
    trans_weight = {}
    for name in model_05_param_name:
        if 'layer' in name:
            layer_num = int(re.search(r'layers\.(\d+)\.', name).group(1))
            model_05_tempweight = model_05_param_name[name]
            name_trans = name.replace(str(layer_num), str(specific_layer[layer_num]))
            model_3_tempweight = model_3_param_name[name_trans]
        else:
            name_trans = name
            model_05_tempweight = model_05_param_name[name]
            model_3_tempweight = model_3_param_name[name]
        if len(model_3_param_name[name_trans].shape) == 1:
            w = torch.linalg.pinv(model_05_tempweight.unsqueeze(0)) @ model_3_tempweight.unsqueeze(0)
        else:
            if model_3_tempweight.shape[0] > model_3_tempweight.shape[1]:
                B = torch.ones([model_3_tempweight.shape[0], model_05_tempweight.shape[0]]) @ model_05_tempweight
                w = torch.linalg.pinv(B) @ model_3_tempweight  ## right_w
            else:
                B = model_05_tempweight @ torch.ones([model_05_tempweight.shape[1], model_3_tempweight.shape[1]])
                w = model_3_tempweight @ torch.linalg.pinv(B)  ## left_w
        trans_weight[name_trans] = w
    return trans_weight


def lm_head_trans_weight(llm_lm_head, slm_lm_head):
    if llm_lm_head.shape[0] == slm_lm_head.shape[0]:
        lm_head_w = torch.linalg.pinv(slm_lm_head) @ llm_lm_head
    else:
        B = torch.ones([llm_lm_head.shape[0], slm_lm_head.shape[0]]) @ slm_lm_head
        lm_head_w = torch.linalg.pinv(B) @ llm_lm_head
    return lm_head_w

"""
## Use for llm transweight_acquire
path_slm = '/data/yakun_data/A-mem/models/Qwen2.5-0.5B-Instruct/'  ## 24 layers
# path_llm = '/root/data/A-mem/models/Qwen2.5-7B-Instruct/'  ## 28 layers
path_llm = '/data/yakun_data/A-mem/models/Qwen2.5-3B-Instruct/'  ## 36 layers
# path_llm = '/data/yakun_data/A-mem/models/Qwen2.5-7B-Instruct/'
# path_slm = '/data/yakun_data/A-mem/models/phi-1_5'  ## 24 layers
# path_llm = '/data/yakun_data/A-mem/models/phi-2'  ## 32 layers
# path_slm = '/data/yakun_data/A-mem/models/Llama-3.2-1B-Instruct/'  ## 16 layers 
# path_llm = '/data/yakun_data/A-mem/models/Llama-3.2-3B-Instruct'  ## 28 layers
model_llm = AutoModelForCausalLM.from_pretrained(path_llm)
model_slm = AutoModelForCausalLM.from_pretrained(path_slm)
model_slm_param_name = {name:param for name, param in model_slm.named_parameters() if param.requires_grad}  ## no lm_head
model_llm_param_name = {name:param for name, param in model_llm.named_parameters() if param.requires_grad}
## layers of slm less than that of llm
# trans_weight = trans_weight_lastlayers(model_slm_param_name, model_llm_param_name, layer_site=4)
trans_weight = trans_weight_lastlayers(model_slm_param_name, model_llm_param_name, layer_site=12)  
## layers of slm over than that of llm
# trans_weight = trans_weight_lastlayers_llm(model_slm_param_name, model_llm_param_name, layer_site=12)  

# ### specific_layer = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35]
# ### trans_weight = trans_weight_gradscore_layer(model_05_param_name, model_3_param_name, specific_layer=None)
# torch.save(trans_weight, "kj_weight/chebi20/llama/llama321B23B_transweight_lastlayers_kj.pth")
torch.save(trans_weight, "kj_weight/chebi20/qwen/qwen2505B23B_transweight_lastlayers_kj.pth")

# lm_head_w = torch.linalg.pinv(model_slm.lm_head.weight) @ model_llm.lm_head.weight  ## only for the same tokens 
lm_head_w = lm_head_trans_weight(model_llm.lm_head.weight, model_slm.lm_head.weight)
torch.save(lm_head_w, "kj_weight/chebi20/qwen/qwen2505B23B_transweight_lmhead_kj.pth")
quit()
"""


def trans_knowledge(model_05, model_05_finetune, model_3_param_name, trans_weight, trans_lm_head_path, layer_site):
    delta_weight_lmhead = model_05_finetune.lm_head.weight - model_05.lm_head.weight
    trans_lmhead = torch.load(trans_lm_head_path) 
    delta_weight_trans_lmhead = delta_weight_lmhead @ trans_lmhead
    delta_weight_trans = {}
    for (name, param), (name_finetune, param_finetune) in zip(model_05.named_parameters(), model_05_finetune.named_parameters()):
        ### last 12 layer
        if 'layers' in name:
            layer_num = int(re.search(r'layers\.(\d+)\.', name).group(1))
            name = name.replace(str(layer_num), str(layer_num+layer_site), 1)
        if param.requires_grad:
            delta_weight = param_finetune - param
            if len(param.shape) == 1:
                trans_weight_delta = (delta_weight.unsqueeze(0) @ trans_weight[name]).squeeze()
            else:
                if model_3_param_name[name].shape[0] > model_3_param_name[name].shape[1]:
                    B = (torch.ones([model_3_param_name[name].shape[0], delta_weight.shape[0]]) @ delta_weight)
                    # B = (torch.eye(model_3_param_name[name].shape[0], delta_weight.shape[0]) @ delta_weight)
                    trans_weight_delta = B @ trans_weight[name]
                else:
                    B = (delta_weight @ torch.ones([delta_weight.shape[1], model_3_param_name[name].shape[1]]))
                    # B = (delta_weight @ torch.eye(delta_weight.shape[1], model_3_param_name[name].shape[1]))
                    trans_weight_delta = trans_weight[name] @ B
            delta_weight_trans[name] = trans_weight_delta
    return delta_weight_trans, delta_weight_trans_lmhead


def trans_knowledge_llm(model_05, model_05_finetune, model_3_param_name, trans_weight, trans_lm_head_path, layer_site):
    delta_weight_lmhead = model_05_finetune.lm_head.weight - model_05.lm_head.weight
    trans_lmhead = torch.load(trans_lm_head_path) 
    delta_weight_trans_lmhead = delta_weight_lmhead @ trans_lmhead
    delta_weight_trans = {}

    for (_, param), (name, param_finetune) in zip(model_05.named_parameters(), model_05_finetune.named_parameters()):
        ### last 12 layer
        if 'layers' in name:
            layer_num = int(re.search(r'layers\.(\d+)\.', name).group(1))
            name_trans = name.replace(str(layer_num), str(layer_num-layer_site), 1)
            if layer_num >= layer_site:
                if param.requires_grad:
                    delta_weight = param_finetune - param
                    if len(param.shape) == 1:
                        trans_weight_delta = (delta_weight.unsqueeze(0) @ trans_weight[name_trans]).squeeze()
                    else:
                        if model_3_param_name[name_trans].shape[0] > model_3_param_name[name_trans].shape[1]:
                            B = (torch.ones([model_3_param_name[name_trans].shape[0], delta_weight.shape[0]]) @ delta_weight)
                            # B = (torch.eye(model_3_param_name[name].shape[0], delta_weight.shape[0]) @ delta_weight)
                            trans_weight_delta = B @ trans_weight[name_trans]
                        else:
                            B = (delta_weight @ torch.ones([delta_weight.shape[1], model_3_param_name[name_trans].shape[1]]))
                            # B = (delta_weight @ torch.eye(delta_weight.shape[1], model_3_param_name[name].shape[1]))
                            trans_weight_delta = trans_weight[name_trans] @ B
                    delta_weight_trans[name_trans] = trans_weight_delta
        elif "lm_head" not in name:
            if param.requires_grad:
                delta_weight = param_finetune - param
                if len(param.shape) == 1:
                    trans_weight_delta = (delta_weight.unsqueeze(0) @ trans_weight[name]).squeeze()
                else:
                    if model_3_param_name[name].shape[0] > model_3_param_name[name].shape[1]:
                        B = (torch.ones([model_3_param_name[name].shape[0], delta_weight.shape[0]]) @ delta_weight)
                        # B = (torch.eye(model_3_param_name[name].shape[0], delta_weight.shape[0]) @ delta_weight)
                        trans_weight_delta = B @ trans_weight[name]
                    else:
                        B = (delta_weight @ torch.ones([delta_weight.shape[1], model_3_param_name[name].shape[1]]))
                        # B = (delta_weight @ torch.eye(delta_weight.shape[1], model_3_param_name[name].shape[1]))
                        trans_weight_delta = trans_weight[name] @ B
                delta_weight_trans[name] = trans_weight_delta
    return delta_weight_trans, delta_weight_trans_lmhead


def trans_knowledge_specific_layer(model_05, model_05_finetune, model_3_param_name, trans_weight, specific_layer):
    delta_weight_lmhead = model_05_finetune.lm_head.weight - model_05.lm_head.weight
    trans_lmhead = torch.load("weight_trans_lmhead.pth")
    delta_weight_trans_lmhead = delta_weight_lmhead @ trans_lmhead
    delta_weight_trans = {}
    for (name, param), (name_finetune, param_finetune) in zip(model_05.named_parameters(), model_05_finetune.named_parameters()):
        if 'layers' in name:
            layer_num = int(re.search(r'layers\.(\d+)\.', name).group(1))
            name = name.replace(str(layer_num), str(specific_layer[layer_num]), 1)
        if param.requires_grad:
            delta_weight = param_finetune - param
            if len(param.shape) == 1:
                trans_weight_delta = (delta_weight.unsqueeze(0) @ trans_weight[name]).squeeze()
            else:
                if model_3_param_name[name].shape[0] > model_3_param_name[name].shape[1]:
                    B = (torch.ones([model_3_param_name[name].shape[0], delta_weight.shape[0]]) @ delta_weight)
                    trans_weight_delta = B @ trans_weight[name]
                else:
                    B = (delta_weight @ torch.ones([delta_weight.shape[1], model_3_param_name[name].shape[1]]))
                    trans_weight_delta = trans_weight[name] @ B
            delta_weight_trans[name] = trans_weight_delta
    return delta_weight_trans, delta_weight_trans_lmhead


def svd_trans_weight(delta_weight_trans, weight_trans_delta_lmhead):
    Uhead, Shead, Vhhead = torch.svd(weight_trans_delta_lmhead)
    r = 8
    Uhead_r = Uhead[:, :r]          
    Shead_r = torch.diag(Shead[:r]) 
    Vhhead_r = Vhhead[:, :r]        
    svd_head = (Uhead_r @ Shead_r, Vhhead_r)
    svd_delta_weight_trans = {}
    for name in delta_weight_trans:
        if len(delta_weight_trans[name].shape) == 2:
            U, S, Vh = torch.svd(delta_weight_trans[name])
            r = 8
            U_r = U[:, :r]          
            S_r = torch.diag(S[:r]) 
            Vh_r = Vh[:, :r]        
            svd_delta_weight_trans[name] = (U_r @ S_r, Vh_r)
    return svd_head, svd_delta_weight_trans


path_slm = '/data/yakun_data/A-mem/models/Qwen2.5-0.5B-Instruct/'  ## 24 layers
# path_llm = '/data/yakun_data/A-mem/models/Qwen2.5-7B-Instruct/'  ## 28 layers
path_llm = '/data/yakun_data/A-mem/models/Qwen2.5-3B-Instruct/'  ## 32 layers
# path_llm = '/data/yakun_data/A-mem/models/Qwen2.5-7B-Instruct/'
# path_llm  = '/data/yakun_data/A-mem/models/phi-2'
# path_slm = '/data/yakun_data/A-mem/models/phi-1_5'
# path_llm = '/data/yakun_data/A-mem/models/Llama-3.2-3B-Instruct/'
# path_slm = '/data/yakun_data/A-mem/models/Llama-3.2-1B-Instruct/'


slm_all_finetune_path = "/data/yakun_data/kj/LLaMA-Factory/saves/medcasereasoning/full/sft/qwen2505B_medcasereasoning_all_epoch6"
# slm_val_finetune_path = "/root/data/kj/LLaMA-Factory/saves/medcasereasoning/full/sft/qwen2505B_medcasereasoning_val_epoch6"
# slm_all_finetune_path = "/data/yakun_data/kj/LLaMA-Factory/saves/medcasereasoning/full/sft/qwen253B_medcasereasoning_all_epoch6_freeze_firstlayers"
# slm_val_finetune_path = '/data/yakun_data/kj/LLaMA-Factory/saves/medcasereasoning/full/sft/qwen253B_medcasereasoning_val_epoch6_freeze_firstlayers'

# slm_all_finetune_path = "/data/yakun_data/kj/LLaMA-Factory/saves/CHEBI20/full/sft/qwen2505b_all_6epoch"
# slm_val_finetune_path = "/data/yakun_data/kj/LLaMA-Factory/saves/CHEBI20/full/sft/qwen2505b_val_6epoch"
# slm_all_finetune_path = '/data/yakun_data/kj/LLaMA-Factory/saves/CHEBI20/full/sft/qwen253b_all_6epoch_freezefirst8layers'
# slm_val_finetune_path = '/data/yakun_data/kj/LLaMA-Factory/saves/CHEBI20/full/sft/qwen253b_val_6epoch_freezefirst8layers'
# slm_all_finetune_path = '/data/yakun_data/kj/LLaMA-Factory/saves/CHEBI20/full/sft/phi15_all_6epoch_sft'
# slm_val_finetune_path = '/data/yakun_data/kj/LLaMA-Factory/saves/CHEBI20/full/sft/phi15_val_6epoch_sft'
# slm_all_finetune_path = '/data/yakun_data/kj/LLaMA-Factory/saves/CHEBI20/full/sft/llama321B_all_6epoch'
# slm_val_finetune_path = '/data/yakun_data/kj/LLaMA-Factory/saves/CHEBI20/full/sft/llama321B_val_6epoch'

# slm_val_finetune_path = '/data/yakun_data/kj/LLaMA-Factory/saves/CHEBI20/full/sft/qwen2505B_sft_6epoch_CHEBI20_001sample'
# slm_val_finetune_path = '/data/yakun_data/kj/LLaMA-Factory/saves/CHEBI20/full/sft/qwen2505B_sft_6epoch_CHEBI20_005sample'
slm_val_finetune_path = '/data/yakun_data/kj/LLaMA-Factory/saves/medcasereasoning/full/sft/qwen2505B_sft_6epoch_medcasereasoning_reasoningdiagnostic_01sample'

# trans_weight_path = "kj_weight/medcasereasoning/qwen2505B27B_transweight_lastlayers_kj.pth"
# trans_lm_head_path = "kj_weight/medcasereasoning/qwen2505B27B_transweight_lmhead_kj.pth"
# trans_weight_path = "kj_weight/medcasereasoning/qwen253B27B_transweight_lastlayers_kj.pth"
# trans_lm_head_path = "kj_weight/medcasereasoning/qwen253B27B_transweight_lmhead_kj.pth"

# trans_weight_path = '/data/yakun_data/kj/kj_weight/chebi20/phi/phi1522_transweight_lastlayers_kj.pth'
# trans_lm_head_path = '/data/yakun_data/kj/kj_weight/chebi20/phi/phi1522_transweight_lmhead_kj.pth'
# trans_weight_path = '/data/yakun_data/kj/kj_weight/chebi20/llama/llama321B23B_transweight_lastlayers_kj.pth'
# trans_lm_head_path = '/data/yakun_data/kj/kj_weight/chebi20/llama/llama321B23B_transweight_lmhead_kj.pth'
trans_weight_path = '/data/yakun_data/kj/kj_weight/chebi20/qwen/qwen2505B23B_transweight_lastlayers_kj.pth'
trans_lm_head_path = '/data/yakun_data/kj/kj_weight/chebi20/qwen/qwen2505B23B_transweight_lmhead_kj.pth'

model_llm = AutoModelForCausalLM.from_pretrained(path_llm)
model_slm = AutoModelForCausalLM.from_pretrained(path_slm)
# model_slm_finetune = AutoModelForCausalLM.from_pretrained(slm_all_finetune_path)
model_slm_finetune_valdata = AutoModelForCausalLM.from_pretrained(slm_val_finetune_path)

trans_weight = torch.load(trans_weight_path)
model_llm_param_name = {name:param for name, param in model_llm.named_parameters() if param.requires_grad}

### qwen2505B to 7B
# delta_weight_trans, delta_weight_trans_lmhead = trans_knowledge(model_slm, model_slm_finetune_valdata, model_llm_param_name, trans_weight, trans_lm_head_path, layer_site=4)
# delta_weight_trans, delta_weight_trans_lmhead = trans_knowledge(model_slm, model_slm_finetune, model_llm_param_name, trans_weight, trans_lm_head_path, layer_site=4)

delta_weight_trans, delta_weight_trans_lmhead = trans_knowledge(model_slm, model_slm_finetune_valdata, model_llm_param_name, trans_weight, trans_lm_head_path, layer_site=12)
# delta_weight_trans, delta_weight_trans_lmhead = trans_knowledge(model_slm, model_slm_finetune, model_llm_param_name, trans_weight, trans_lm_head_path, layer_site=12)


### qwen253b to 7B
# delta_weight_trans, delta_weight_trans_lmhead = trans_knowledge_llm(model_slm, model_slm_finetune_valdata, model_llm_param_name, trans_weight, trans_lm_head_path, layer_site=8)
# delta_weight_trans, delta_weight_trans_lmhead = trans_knowledge_llm(model_slm, model_slm_finetune, model_llm_param_name, trans_weight, trans_lm_head_path, layer_site=8)


### Specific Layer
# specific_layer = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35]
# delta_weight_trans, delta_weight_trans_lmhead = trans_knowledge_specific_layer(model_05, model_05_finetune_valdata, model_3_param_name, trans_weight, specific_layer)


# torch.save(delta_weight_trans, "kj_weight/medcasereasoning/valdata_kj_lastlayers.pth")
# torch.save(delta_weight_trans_lmhead, "kj_weight/medcasereasoning/valdata_kj_lastlayers_lmhead.pth")
# torch.save(delta_weight_trans, "kj_weight/medcasereasoning/qwen253B27B_alldata_kj_lastlayers.pth")
# torch.save(delta_weight_trans_lmhead, "kj_weight/medcasereasoning/qwen253B27B_alldata_kj_lastlayers_lmhead.pth")

# torch.save(delta_weight_trans, "kj_weight/chebi20/qwen2505B27B_alldata_kj_lastlayers.pth")
# torch.save(delta_weight_trans_lmhead, "kj_weight/chebi20/qwen2505B27B_alldata_kj_lastlayers_lmhead.pth")
# torch.save(delta_weight_trans, 'kj_weight/chebi20/qwen253B27B_valdata_kj_lastlayers.pth')
# torch.save(delta_weight_trans_lmhead, 'kj_weight/chebi20/qwen253B27B_valdata_kj_lastlayers_lmhead.pth')
# torch.save(delta_weight_trans, 'kj_weight/chebi20/llama/llama321B23B_valdata_kj_lastlayers.pth')
# torch.save(delta_weight_trans_lmhead, 'kj_weight/chebi20/llama/llama321B23B_valdata_kj_lastlayers_lmhead.pth')
# torch.save(delta_weight_trans, 'kj_weight/medcasereasoning/qwen/qwen2505B23B_alldata_kj_lastlayers.pth')
# torch.save(delta_weight_trans_lmhead, 'kj_weight/medcasereasoning/qwen/qwen2505B23B_alldata_lastlayers_lmhead.pth')
torch.save(delta_weight_trans, 'kj_weight/medcasereasoning/qwen/qwen2505B23B_01sample_valdata_kj_lastlayers.pth')
torch.save(delta_weight_trans_lmhead, 'kj_weight/medcasereasoning/qwen/qwen2505B23B_01sample_valdata_kj_lastlayers_lmhead.pth')


quit()
quit()
quit()



##### SVD
weight_trans_delta_lmhead = torch.load('kj_weight/medcasereasoning/valdata_kj_lastlayers_lmhead.pth')
delta_weight_trans = torch.load('kj_weight/medcasereasoning/valdata_kj_lastlayers.pth')
svd_head, svd_delta_weight_trans = svd_trans_weight(delta_weight_trans, weight_trans_delta_lmhead)
torch.save(svd_head, "kj_weight/medcasereasoning/svd_valdata_kj_lastlayers_lmhead.pth")
torch.save(svd_delta_weight_trans, "kj_weight/medcasereasoning/svd_valdata_kj_lastlayers.pth")
quit()
quit()
quit()
