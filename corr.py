from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import os,pylab,torch,scipy

llms = {
    # "google/gemma-3-4b-it": [34,8,256,2],
    # "google/gemma-3-1b-it": [26,4,256,4],
    # "google/gemma-2-2b": [26,8,256,2],
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B":[28,12,128,6],
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B":[32,32,128,4],
    # "google/gemma-2-2b-it": [26,8,256,2],
    # "meta-llama/Llama-3.2-1B-Instruct": [16,32,64,4],    
    # "meta-llama/Llama-3.2-3B-Instruct": [28,24,128,3],
    # "Qwen/Qwen2.5-0.5B-Instruct": [24,14,64,7],
    # "Qwen/Qwen2.5-1.5B-Instruct": [28,12,128,6],
    # "mistralai/Mistral-7B-Instruct-v0.3": [32,32,128,4],
    # "google/gemma-2-9b-it": [42,16,256,2],
    # "google/gemma-2b-it": [18,8,256,8],    
    # "google/gemma-7b-it": [28,16,256,1],
    # "Qwen/Qwen2.5-7B-Instruct": [28,28,128,7],
    # "google/gemma-2-27b": [46,32,128,2],
    # "meta-llama/Llama-3.3-70B-Instruct": [80,64,128,8],
    # "meta-llama/Meta-Llama-3.1-8B-Instruct": [32,32,128,4],
    # "meta-llama/Meta-Llama-3.1-8B": [32,32,128,4],
    # "meta-llama/Meta-Llama-3-8B-Instruct": [32,32,128,4],
    # "meta-llama/Meta-Llama-3-8B": [32,32,128,4],
    # "google/gemma-2b": [18,8,256,8],
    # "google/gemma-7b": [28,16,256,1],
    # "google/gemma-2-9b": [42,16,256,2],
    # "mistralai/Mistral-7B-v0.3": [32,32,128,4],
    # "meta-llama/Llama-3.2-1B": [16,32,64,4],
    # "meta-llama/Llama-3.2-3B": [28,24,128,3],
    # "Qwen/Qwen2.5-0.5B": [24,14,64,7],
    # "Qwen/Qwen2.5-1.5B": [28,12,128,6],
    # "Qwen/Qwen2.5-7B": [28,28,128,7],
}
    
def get_weights(b=0,l='self_attn',p='q_proj',h=None,hdim=256,hratio=2,model=None):
    if p:
        # key = 'model.language_model.layers.'+str(b)+'.'+str(l)+'.'+str(p)+'.weight'
        key = 'model.layers.'+str(b)+'.'+str(l)+'.'+str(p)+'.weight'
    elif l and b != None:
        # key = 'model.language_model.layers.'+str(b)+'.'+str(l)+'.weight' 
        key = 'model.layers.'+str(b)+'.'+str(l)+'.weight'
    elif l:
        key = 'model.'+str(l)+'.weight'
    else:
        key = 'lm_head.weight'
    if h != None and l == 'self_attn':
        if p[0] == 'q':
            mat = model.state_dict()[key][h*hdim:(h+1)*hdim,:]
        elif p[0] == 'k' or p[0] == 'v':
            mat = model.state_dict()[key][(h//hratio)*hdim:(h//hratio+1)*hdim,:]
        elif p[0] == 'o':
            mat = model.state_dict()[key][:,h*hdim:(h+1)*hdim]
    else:
        mat = model.state_dict()[key]
    mat = mat.to(torch.float)
    if model.device.type == 'cpu':
        mat = mat.cpu()
    return mat.numpy(),key
    
def print_fig(nb=26,nh=8,hdim=256,hratio=2,model=None):
    wq_wk = np.empty((nb,nh))
    wq_wv = np.empty((nb,nh))
    wk_wv = np.empty((nb,nh))
    wo_wv = np.empty((nb,nh))
    wqwk_eye = np.empty((nb,nh))
    # wowv_eye = np.empty((nb,nh))
    ww = np.empty((nb,4))
    ww_mlp = np.empty((nb,3))
    for b in range(nb):
        print(b)
        for h in range(nh):
            wq,_ = get_weights(b=b,l='self_attn',p='q_proj',h=h,hdim=hdim,hratio=hratio,model=model)
            wk,_ = get_weights(b=b,l='self_attn',p='k_proj',h=h,hdim=hdim,hratio=hratio,model=model)
            wv,_ = get_weights(b=b,l='self_attn',p='v_proj',h=h,hdim=hdim,hratio=hratio,model=model)
            wo,_ = get_weights(b=b,l='self_attn',p='o_proj',h=h,hdim=hdim,hratio=hratio,model=model)   
            c1 = np.corrcoef(wq.flatten(),wk.flatten())[0,1]
            c2 = np.corrcoef(wq.flatten(),wv.flatten())[0,1]
            c3 = np.corrcoef(wk.flatten(),wv.flatten())[0,1]
            c4 = np.corrcoef(np.matmul(wq.T,wk).flatten(),np.eye(wq.shape[1]).flatten())[0,1]
            # c6 = np.corrcoef(np.matmul(wo,wv).flatten(),np.eye(wo.shape[0]).flatten())[0,1]
            wo = wo.T
            c5 = np.corrcoef(wo.flatten(),wv.flatten())[0,1]
            wq_wk[b,h] = c1
            wq_wv[b,h] = c2
            wk_wv[b,h] = c3
            wqwk_eye[b,h] = c4
            wo_wv[b,h] = c5
            # wowv_eye[b,h] = c6
        wg,_ = get_weights(b=b,l='mlp',p='gate_proj',model=model)
        wu,_ = get_weights(b=b,l='mlp',p='up_proj',model=model)
        wd,_ = get_weights(b=b,l='mlp',p='down_proj',model=model)
        iwu = np.linalg.pinv(wu)
        c4 = np.corrcoef(iwu.flatten(),wd.flatten())[0,1]
        c5 = np.corrcoef(np.matmul(wd,wg).flatten(),np.eye(wg.shape[1]).flatten())[0,1]
        c6 = np.corrcoef(np.matmul(wd,wu).flatten(),np.eye(wg.shape[1]).flatten())[0,1]
        c7 = np.corrcoef(np.matmul(wd,iwu.T).flatten(),np.eye(wg.shape[1]).flatten())[0,1]
        wd = wd.T
        c1 = np.corrcoef(wg.flatten(),wu.flatten())[0,1]
        c2 = np.corrcoef(wg.flatten(),wd.flatten())[0,1]
        c3 = np.corrcoef(wu.flatten(),wd.flatten())[0,1]
        ww[b,0] = c1
        ww[b,1] = c2
        ww[b,2] = c3
        ww[b,3] = c4
        ww_mlp[b,0] = c5
        ww_mlp[b,1] = c6
        ww_mlp[b,2] = c7
    cmin = np.min(wq_wk) 
    cmax = np.max(wq_wk)
    bnumber = np.array([i for i in range(nb) for _ in range(nh)])
    pylab.figure(figsize=[14,8])
    pylab.subplot(241)
    pylab.imshow(wq_wk) 
    pylab.colorbar()
    pylab.clim([cmin,cmax])
    pylab.title('cc of wq and wk')
    pylab.ylabel('layer')
    pylab.subplot(242)
    pylab.imshow(wq_wv)
    pylab.colorbar()
    pylab.clim([cmin,cmax])
    pylab.title('cc of wq and wv')
    pylab.subplot(243)
    pylab.imshow(wk_wv)
    pylab.colorbar()
    pylab.clim([cmin,cmax])
    pylab.title('cc of wk and wv')
    pylab.subplot(244)
    pylab.imshow(wqwk_eye)
    pylab.colorbar()
    pylab.title('cc of wqTwk and I')
    pylab.subplot(245)
    pylab.imshow(wo_wv)
    pylab.colorbar()
    pylab.title('cc of woT and wv')
    pylab.xlabel('head')
    pylab.ylabel('layer')
    pylab.subplot(246)
    slope,intercept,rvalue,pvalue,_= scipy.stats.linregress(wq_wk.flatten(),wo_wv.flatten())
    x = np.array([np.min(wq_wk),np.max(wq_wk)])
    pylab.plot(x,slope*x+intercept,'k',lw=1)
    pylab.scatter(wq_wk.flatten(),wo_wv.flatten(),c=bnumber,cmap='Greens_r',s=6)
    pylab.text(0,np.max(wo_wv)*0.8,f'cc={rvalue:.2f}')
    pylab.colorbar()
    pylab.xlabel('cc of wq and wk')
    pylab.ylabel('cc of woT and wv')
    pylab.subplot(247)
    pylab.imshow(ww)
    pylab.colorbar()
    pylab.title('cc of w in mlp')
    pylab.xlabel('w pair')
    pylab.subplot(248)
    pylab.imshow(ww_mlp)
    pylab.colorbar()
    pylab.title('cc of ww in mlp and I')
    pylab.subplots_adjust(wspace=0.5)

for llm in llms.keys():
    model = AutoModelForCausalLM.from_pretrained(llm,torch_dtype=torch.bfloat16,device_map="auto")
    nb,nh,hdim,hratio = llms[llm]
    print_fig(nb=nb,nh=nh,hdim=hdim,hratio=hratio,model=model)
    pylab.suptitle(llm)
    pylab.savefig(llm.split('/')[-1]+'.png')
    print(llm)
