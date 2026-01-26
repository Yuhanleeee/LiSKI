import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re


def trans_weight_lastlayers_inverse(model_slm_param_name, model_llm_param_name, layer_site):
    """
    layers of slm over than that of llm, e.g., qwen253b to qwen257b
    """
    trans_weight = {}
    for name in model_llm_param_name:
        if 'lm_head' not in name:
            if 'layers' in name:
                layer_num = int(re.search(r'layers\.(\d+)\.', name).group(1))
                name_trans = name.replace(str(layer_num), str(layer_num+layer_site))
                model_slm_tempweight = model_slm_param_name[name_trans]
                model_llm_tempweight = model_llm_param_name[name]
            else:
                model_slm_tempweight = model_slm_param_name[name]
                model_llm_tempweight = model_llm_param_name[name]
            if len(model_llm_param_name[name].shape) == 1:
                # with .T
                # temp = (model_slm_tempweight.unsqueeze(0).T) @ model_slm_tempweight.unsqueeze(0)
                # w = torch.linalg.pinv(temp) @ (model_slm_tempweight.unsqueeze(0).T @ model_llm_tempweight.unsqueeze(0))
                ## without .T
                w = torch.linalg.pinv(model_slm_tempweight.unsqueeze(0)) @ model_llm_tempweight.unsqueeze(0)
            else:
                if model_llm_tempweight.shape[0] > model_llm_tempweight.shape[1]:  ## right_w
                    ## with .T
                    ### ones for B
                    # onesw = torch.ones([model_slm_tempweight.shape[0], model_llm_tempweight.shape[0]])
                    # w = torch.linalg.pinv(model_slm_tempweight.T @ model_slm_tempweight) @ model_slm_tempweight.T @ torch.linalg.pinv(onesw.T @ onesw) @ onesw.T @ model_llm_tempweight
                    ### eye for B
                    # iw = torch.eye(model_llm_tempweight.shape[0], model_slm_tempweight.shape[0])
                    # w = torch.linalg.pinv(model_slm_tempweight.T @ model_slm_tempweight) @ model_slm_tempweight.T @ iw.T @ model_llm_tempweight
                    ## without .T
                    B = torch.ones([model_llm_tempweight.shape[0], model_slm_tempweight.shape[0]]) @ model_slm_tempweight
                    # B = torch.eye([model_llm_tempweight.shape[0], model_slm_tempweight.shape[0]]) @ model_slm_tempweight
                    w = torch.linalg.pinv(B) @ model_llm_tempweight  ## right_w
                else:  ## left_w
                    ## with .T
                    ### ones for B
                    # onesw = torch.ones([model_slm_tempweight.shape[1], model_llm_tempweight.shape[1]])
                    # w = model_llm_tempweight @ onesw.T @ torch.linalg.pinv(onesw @ onesw.T) @ model_slm_tempweight.T @ torch.linalg.pinv(model_slm_tempweight @ model_slm_tempweight.T)
                    ### eye for B
                    # iw = torch.eye(model_slm_tempweight.shape[1], model_llm_tempweight.shape[1])
                    # w = model_llm_tempweight @ iw.T @ model_slm_tempweight.T @ torch.linalg.pinv(model_slm_tempweight @ model_slm_tempweight.T)
                    ## without .T
                    B = model_slm_tempweight @ torch.ones([model_slm_tempweight.shape[1], model_llm_tempweight.shape[1]])
                    # B = model_slm_tempweight @ torch.eye([model_slm_tempweight.shape[1], model_llm_tempweight.shape[1]])
                    w = model_llm_tempweight @ torch.linalg.pinv(B)  ## left_w
            trans_weight[name] = w
    return trans_weight


def trans_weight_lastlayers(model_slm_param_name, model_llm_param_name, layer_site):
    """
    layers of slm less than that of llm, e.g., qwen2515b to qwen253b
    """
    trans_weight = {}
    for name in model_slm_param_name:
        if 'layers' in name:
            layer_num = int(re.search(r'layers\.(\d+)\.', name).group(1))
            model_slm_tempweight = model_slm_param_name[name]
            name_trans = name.replace(str(layer_num), str(layer_num+layer_site), 1)
            model_llm_tempweight = model_llm_param_name[name_trans]
        else:
            name_trans = name 
            model_slm_tempweight = model_slm_param_name[name]
            model_llm_tempweight = model_llm_param_name[name]
        if len(model_llm_param_name[name_trans].shape) == 1:
            ## with .T
            # temp = (model_slm_tempweight.unsqueeze(0).T) @ model_slm_tempweight.unsqueeze(0)
            # w = torch.linalg.pinv(temp) @ (model_slm_tempweight.unsqueeze(0).T @ model_llm_tempweight.unsqueeze(0))
            ## without .T
            w = torch.linalg.pinv(model_slm_tempweight.unsqueeze(0)) @ model_llm_tempweight.unsqueeze(0)
        else:
            if model_llm_tempweight.shape[0] > model_llm_tempweight.shape[1]:  ## right_w
                ## with .T
                ### ones for B
                # onesw = torch.ones([model_slm_tempweight.shape[0], model_llm_tempweight.shape[0]])
                # w = torch.linalg.pinv(model_slm_tempweight.T @ model_slm_tempweight) @ model_slm_tempweight.T @ torch.linalg.pinv(onesw.T @ onesw) @ onesw.T @ model_llm_tempweight
                ### eye for B
                # iw = torch.eye(model_llm_tempweight.shape[0], model_slm_tempweight.shape[0])
                # w = torch.linalg.pinv(model_slm_tempweight.T @ model_slm_tempweight) @ model_slm_tempweight.T @ iw.T @ model_llm_tempweight
                ## without .T
                B = torch.ones([model_llm_tempweight.shape[0], model_slm_tempweight.shape[0]]) @ model_slm_tempweight
                # B = torch.eye([model_llm_tempweight.shape[0], model_slm_tempweight.shape[0]]) @ model_slm_tempweight
                w = torch.linalg.pinv(B) @ model_llm_tempweight  ## right_w
            else:  ## left_w
                ## with .T
                ### ones for B
                # onesw = torch.ones([model_slm_tempweight.shape[1], model_llm_tempweight.shape[1]])
                # w = model_llm_tempweight @ onesw.T @ torch.linalg.pinv(onesw @ onesw.T) @ model_slm_tempweight.T @ torch.linalg.pinv(model_slm_tempweight @ model_slm_tempweight.T)
                ### eye for B
                # iw = torch.eye(model_slm_tempweight.shape[1], model_llm_tempweight.shape[1])
                # w = model_llm_tempweight @ iw.T @ model_slm_tempweight.T @ torch.linalg.pinv(model_slm_tempweight @ model_slm_tempweight.T)
                ## without .T
                B = model_slm_tempweight @ torch.ones([model_slm_tempweight.shape[1], model_llm_tempweight.shape[1]])
                # B = model_slm_tempweight @ torch.eye([model_slm_tempweight.shape[1], model_llm_tempweight.shape[1]])
                w = model_llm_tempweight @ torch.linalg.pinv(B)  ## left_w
        trans_weight[name_trans] = w
    return trans_weight


def trans_weight_gradscore_layer(model_slm_param_name, model_llm_param_name, specific_layer):
    trans_weight = {}
    for name in model_slm_param_name:
        if 'layer' in name:
            layer_num = int(re.search(r'layers\.(\d+)\.', name).group(1))
            model_slm_tempweight = model_slm_param_name[name]
            name_trans = name.replace(str(layer_num), str(specific_layer[layer_num]))
            model_llm_tempweight = model_llm_param_name[name_trans]
        else:
            name_trans = name
            model_slm_tempweight = model_slm_param_name[name]
            model_llm_tempweight = model_llm_param_name[name]
        if len(model_llm_param_name[name_trans].shape) == 1:
            w = torch.linalg.pinv(model_slm_tempweight.unsqueeze(0)) @ model_llm_tempweight.unsqueeze(0)
        else:
            if model_llm_tempweight.shape[0] > model_llm_tempweight.shape[1]:
                B = torch.ones([model_llm_tempweight.shape[0], model_slm_tempweight.shape[0]]) @ model_slm_tempweight
                # B = torch.eye([model_llm_tempweight.shape[0], model_slm_tempweight.shape[0]]) @ model_slm_tempweight
                w = torch.linalg.pinv(B) @ model_llm_tempweight  ## right_w
            else:
                B = model_slm_tempweight @ torch.ones([model_slm_tempweight.shape[1], model_llm_tempweight.shape[1]])
                # B = model_slm_tempweight @ torch.eye([model_slm_tempweight.shape[1], model_llm_tempweight.shape[1]])
                w = model_llm_tempweight @ torch.linalg.pinv(B)  ## left_w
        trans_weight[name_trans] = w
    return trans_weight


def lm_head_trans_weight(llm_lm_head, slm_lm_head):
    if llm_lm_head.shape[0] == slm_lm_head.shape[0]:
        lm_head_w = torch.linalg.pinv(slm_lm_head) @ llm_lm_head
    else:
        B = torch.ones([llm_lm_head.shape[0], slm_lm_head.shape[0]]) @ slm_lm_head
        # B = torch.eye([llm_lm_head.shape[0], slm_lm_head.shape[0]]) @ slm_lm_head
        lm_head_w = torch.linalg.pinv(B) @ llm_lm_head
    return lm_head_w


def trans_knowledge(model_slm, model_slm_finetune, model_llm_param_name, trans_weight, trans_lm_head_path, layer_site):
    delta_weight_lmhead = model_slm_finetune.lm_head.weight - model_slm.lm_head.weight
    trans_lmhead = torch.load(trans_lm_head_path) 
    delta_weight_trans_lmhead = delta_weight_lmhead @ trans_lmhead
    delta_weight_trans = {}
    for (name, param), (name_finetune, param_finetune) in zip(model_slm.named_parameters(), model_slm_finetune.named_parameters()):
        ### last layers
        if 'layers' in name:
            layer_num = int(re.search(r'layers\.(\d+)\.', name).group(1))
            name = name.replace(str(layer_num), str(layer_num+layer_site), 1)
        if param.requires_grad:
            delta_weight = param_finetune - param
            if len(param.shape) == 1:
                trans_weight_delta = (delta_weight.unsqueeze(0) @ trans_weight[name]).squeeze()
            else:
                if model_llm_param_name[name].shape[0] > model_llm_param_name[name].shape[1]:
                    B = (torch.ones([model_llm_param_name[name].shape[0], delta_weight.shape[0]]) @ delta_weight)
                    # B = (torch.eye(model_llm_param_name[name].shape[0], delta_weight.shape[0]) @ delta_weight)
                    trans_weight_delta = B @ trans_weight[name]
                else:
                    B = (delta_weight @ torch.ones([delta_weight.shape[1], model_llm_param_name[name].shape[1]]))
                    # B = (delta_weight @ torch.eye(delta_weight.shape[1], model_llm_param_name[name].shape[1]))
                    trans_weight_delta = trans_weight[name] @ B
            delta_weight_trans[name] = trans_weight_delta
    return delta_weight_trans, delta_weight_trans_lmhead


def trans_knowledge_inverse(model_slm, model_slm_finetune, model_llm_param_name, trans_weight, trans_lm_head_path, layer_site):
    ## layer numbers of slm over than that of llm
    delta_weight_lmhead = model_slm_finetune.lm_head.weight - model_slm.lm_head.weight
    trans_lmhead = torch.load(trans_lm_head_path) 
    delta_weight_trans_lmhead = delta_weight_lmhead @ trans_lmhead
    delta_weight_trans = {}

    for (_, param), (name, param_finetune) in zip(model_slm.named_parameters(), model_slm_finetune.named_parameters()):
        ### last layers
        if 'layers' in name:
            layer_num = int(re.search(r'layers\.(\d+)\.', name).group(1))
            name_trans = name.replace(str(layer_num), str(layer_num-layer_site), 1)
            if layer_num >= layer_site:
                if param.requires_grad:
                    delta_weight = param_finetune - param
                    if len(param.shape) == 1:
                        trans_weight_delta = (delta_weight.unsqueeze(0) @ trans_weight[name_trans]).squeeze()
                    else:
                        if model_llm_param_name[name_trans].shape[0] > model_llm_param_name[name_trans].shape[1]:
                            B = (torch.ones([model_llm_param_name[name_trans].shape[0], delta_weight.shape[0]]) @ delta_weight)
                            # B = (torch.eye(model_llm_param_name[name].shape[0], delta_weight.shape[0]) @ delta_weight)
                            trans_weight_delta = B @ trans_weight[name_trans]
                        else:
                            B = (delta_weight @ torch.ones([delta_weight.shape[1], model_llm_param_name[name_trans].shape[1]]))
                            # B = (delta_weight @ torch.eye(delta_weight.shape[1], model_llm_param_name[name].shape[1]))
                            trans_weight_delta = trans_weight[name_trans] @ B
                    delta_weight_trans[name_trans] = trans_weight_delta
        elif "lm_head" not in name:
            if param.requires_grad:
                delta_weight = param_finetune - param
                if len(param.shape) == 1:
                    trans_weight_delta = (delta_weight.unsqueeze(0) @ trans_weight[name]).squeeze()
                else:
                    if model_llm_param_name[name].shape[0] > model_llm_param_name[name].shape[1]:
                        B = (torch.ones([model_llm_param_name[name].shape[0], delta_weight.shape[0]]) @ delta_weight)
                        # B = (torch.eye(model_llm_param_name[name].shape[0], delta_weight.shape[0]) @ delta_weight)
                        trans_weight_delta = B @ trans_weight[name]
                    else:
                        B = (delta_weight @ torch.ones([delta_weight.shape[1], model_llm_param_name[name].shape[1]]))
                        # B = (delta_weight @ torch.eye(delta_weight.shape[1], model_llm_param_name[name].shape[1]))
                        trans_weight_delta = trans_weight[name] @ B
                delta_weight_trans[name] = trans_weight_delta
    return delta_weight_trans, delta_weight_trans_lmhead


def trans_knowledge_specific_layer(model_slm, model_slm_finetune, model_llm_param_name, trans_weight, specific_layer):
    delta_weight_lmhead = model_slm_finetune.lm_head.weight - model_slm.lm_head.weight
    trans_lmhead = torch.load("weight_trans_lmhead.pth")
    delta_weight_trans_lmhead = delta_weight_lmhead @ trans_lmhead
    delta_weight_trans = {}
    for (name, param), (name_finetune, param_finetune) in zip(model_slm.named_parameters(), model_slm_finetune.named_parameters()):
        if 'layers' in name:
            layer_num = int(re.search(r'layers\.(\d+)\.', name).group(1))
            name = name.replace(str(layer_num), str(specific_layer[layer_num]), 1)
        if param.requires_grad:
            delta_weight = param_finetune - param
            if len(param.shape) == 1:
                trans_weight_delta = (delta_weight.unsqueeze(0) @ trans_weight[name]).squeeze()
            else:
                if model_llm_param_name[name].shape[0] > model_llm_param_name[name].shape[1]:
                    B = (torch.ones([model_llm_param_name[name].shape[0], delta_weight.shape[0]]) @ delta_weight)
                    # B = (torch.eye([model_llm_param_name[name].shape[0], delta_weight.shape[0]]) @ delta_weight)
                    trans_weight_delta = B @ trans_weight[name]
                else:
                    B = (delta_weight @ torch.ones([delta_weight.shape[1], model_llm_param_name[name].shape[1]]))
                    # B = (delta_weight @ torch.eye([delta_weight.shape[1], model_llm_param_name[name].shape[1]]))
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



## Use for llm transweight_acquire
path_slm = 'models/Qwen2.5-0.5B-Instruct/'  ## 24 layers
# path_llm = 'models/Qwen2.5-7B-Instruct/'  ## 28 layers
path_llm = 'models/Qwen2.5-3B-Instruct/'  ## 36 layers
# path_slm = 'models/phi-1_5'  ## 24 layers
# path_llm = 'models/phi-2'  ## 32 layers
# path_slm = 'models/Llama-3.2-1B-Instruct/'  ## 16 layers 
# path_llm = 'models/Llama-3.2-3B-Instruct'  ## 28 layers
model_llm = AutoModelForCausalLM.from_pretrained(path_llm)
model_slm = AutoModelForCausalLM.from_pretrained(path_slm)
model_slm_param_name = {name:param for name, param in model_slm.named_parameters() if param.requires_grad}  ## no lm_head
model_llm_param_name = {name:param for name, param in model_llm.named_parameters() if param.requires_grad}
## layers of slm less than that of llm
# trans_weight = trans_weight_lastlayers(model_slm_param_name, model_llm_param_name, layer_site=4)
trans_weight = trans_weight_lastlayers(model_slm_param_name, model_llm_param_name, layer_site=12)  
## layers of slm over than that of llm
# trans_weight = trans_weight_lastlayers_inverse(model_slm_param_name, model_llm_param_name, layer_site=12)  

# ### specific_layer = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35]
# ### trans_weight = trans_weight_gradscore_layer(model_slm_param_name, model_llm_param_name, specific_layer=None)
torch.save(trans_weight, "kj_weight/.pth")

# lm_head_w = torch.linalg.pinv(model_slm.lm_head.weight) @ model_llm.lm_head.weight  
lm_head_w = lm_head_trans_weight(model_llm.lm_head.weight, model_slm.lm_head.weight)
torch.save(lm_head_w, "kj_weight_lmhead/.pth")


path_slm = 'models/Qwen2.5-0.5B-Instruct/'  ## 24 layers
# path_llm = 'models/Qwen2.5-7B-Instruct/'  ## 28 layers
path_llm = 'models/Qwen2.5-3B-Instruct/'  ## 32 layers
# path_llm = 'models/Qwen2.5-7B-Instruct/'
# path_llm  = 'models/phi-2'
# path_slm = 'models/phi-1_5'
# path_llm = 'models/Llama-3.2-3B-Instruct/'
# path_slm = 'models/Llama-3.2-1B-Instruct/'


slm_all_finetune_path = ""  ## fine-tuned slm with all domain dataset
slm_val_finetune_path = ""  ## fine-tuned slm with calibration dataset

trans_weight_path = 'kj_weihgt/.pth'
trans_lm_head_path = 'kj_weight_lmhead/.pth'

model_llm = AutoModelForCausalLM.from_pretrained(path_llm)
model_slm = AutoModelForCausalLM.from_pretrained(path_slm)
model_slm_finetune = AutoModelForCausalLM.from_pretrained(slm_all_finetune_path)
model_slm_finetune_valdata = AutoModelForCausalLM.from_pretrained(slm_val_finetune_path)

trans_weight = torch.load(trans_weight_path)
model_llm_param_name = {name:param for name, param in model_llm.named_parameters() if param.requires_grad}


## knowledge injection
delta_weight_trans_val, delta_weight_trans_lmhead_val = trans_knowledge(model_slm, model_slm_finetune_valdata, model_llm_param_name, trans_weight, trans_lm_head_path, layer_site=12)
delta_weight_trans, delta_weight_trans_lmhead = trans_knowledge(model_slm, model_slm_finetune, model_llm_param_name, trans_weight, trans_lm_head_path, layer_site=12)


### knowledge injection, layer numbers of sml over than that of llm
# delta_weight_trans_val, delta_weight_trans_lmhead_val = trans_knowledge_inverse(model_slm, model_slm_finetune_valdata, model_llm_param_name, trans_weight, trans_lm_head_path, layer_site=8)
# delta_weight_trans, delta_weight_trans_lmhead = trans_knowledge_inverse(model_slm, model_slm_finetune, model_llm_param_name, trans_weight, trans_lm_head_path, layer_site=8)


### Specific Layer
# specific_layer = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35]
# delta_weight_trans, delta_weight_trans_lmhead = trans_knowledge_specific_layer(model_slm, model_slm_finetune_valdata, model_llm_param_name, trans_weight, specific_layer)


torch.save(delta_weight_trans, 'kj_weight_all/.pth')
torch.save(delta_weight_trans_lmhead, 'kj_weight_lmhead_all/.pth')

torch.save(delta_weight_trans_val, 'kj_weight_calibration/.pth')
torch.save(delta_weight_trans_lmhead_val, 'kj_weight_lmhead_calibration/.pth')

"""
##### SVD for space efficiency
delta_weight_trans_lmhead = torch.load('kj_weight_lmhead/.pth')
delta_weight_trans = torch.load('kj_weight/.pth')
svd_head, svd_delta_weight_trans = svd_trans_weight(delta_weight_trans, delta_weight_trans_lmhead)
torch.save(svd_head, "kj_weight_lmhead_svd/.pth")
torch.save(svd_delta_weight_trans, "kj_weight_svd/.pth")
"""
