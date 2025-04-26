#!/usr/bin/env python3
"""
Generate all CSV tables and figures (1–5) for living/dead and type comparisons
on random vs quota datasets, unconditioned vs conditioned.
Plots use Times New Roman for a publication-ready style.
"""
import os, sys, glob, csv, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

# publication style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# repo path
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, repo_root)

from models.unet import UNet
from models.diffusion import Diffusion
from phase_2_conditional_diffusion.label_training_data_32x32 import classify_grid

# categories
cats_types = ['died_out', 'still_life', 'oscillator_period_2', 'others']
# living/dead mapping
cats_ld = ['live', 'dead']

# map raw to types
fmap = lambda raw: raw if raw in cats_types else 'others'

# compute proportions for dataset
def compute_dataset(data_dir, timesteps, types=False):
    files = glob.glob(os.path.join(data_dir,'**','*.npy'), recursive=True)
    cnt = {}
    if types:
        cnt = {c:0 for c in cats_types}
        for p in files:
            arr = np.load(p).astype(np.uint8)
            key = fmap(classify_grid(arr, timesteps))
            cnt[key] += 1
    else:
        cnt = {'live':0,'dead':0}
        for p in files:
            arr = np.load(p).astype(np.uint8)
            cat = classify_grid(arr, timesteps)
            cnt['dead' if cat=='died_out' else 'live'] += 1
    total = sum(cnt.values())
    return {k: cnt[k] for k in cnt}, {k: cnt[k]/total*100 for k in cnt}

# sample from model
def sample_model(model, diff, device, num, timesteps, condition, types=False):
    c = None if condition is None else torch.full((num,), condition, dtype=torch.long, device=device)
    with torch.no_grad(): x = diff.sample(model, shape=(num,1,32,32), c=c)
    arrs = (x.cpu().numpy()>0.5).astype(np.uint8).squeeze(1)
    cnt = {}
    if types:
        cnt = {c:0 for c in cats_types}
        for arr in arrs:
            key = fmap(classify_grid(arr, timesteps))
            cnt[key] += 1
    else:
        cnt = {'live':0,'dead':0}
        for arr in arrs:
            cat = classify_grid(arr, timesteps)
            cnt['dead' if cat=='died_out' else 'live'] += 1
    return cnt, {k: cnt[k]/num*100 for k in cnt}

# orchestrate one scenario
def evaluate(data_dir, ckpt, timesteps, num, output_prefix):
    print(f"[generate_report] Starting evaluation for '{output_prefix}'")
    # dataset
    print(f"[generate_report] {output_prefix}: computing living/dead stats...")
    d_cnt, d_pct = compute_dataset(data_dir,timesteps,types=False)
    print(f"[generate_report] {output_prefix}: computing type stats...")
    d_cnt_t, d_pct_t = compute_dataset(data_dir,timesteps,types=True)
    print(f"[generate_report] {output_prefix}: loading model from {ckpt}...")
    # load model
    model = UNet(dropout=0.0,num_classes=2).to(args.device)
    state = torch.load(ckpt,map_location=args.device)
    model.load_state_dict(state); model.eval()
    print(f"[generate_report] {output_prefix}: sampling uncond and types...")
    diff = Diffusion(timesteps=timesteps,device=args.device,
                     schedule=args.schedule,guidance_scale=args.guidance_scale)
    # samples
    uc_cnt, uc_pct = sample_model(model,diff,args.device,num,timesteps,None,types=False)
    uc_cnt_t, uc_pct_t = sample_model(model,diff,args.device,num,timesteps,None,types=True)
    print(f"[generate_report] {output_prefix}: sampling cond live...")
    cl_cnt, cl_pct = sample_model(model,diff,args.device,num,timesteps,1,types=False)
    print(f"[generate_report] {output_prefix}: sampling cond dead...")
    cd_cnt, cd_pct = sample_model(model,diff,args.device,num,timesteps,0,types=False)
    print(f"[generate_report] {output_prefix}: writing CSVs and plotting figures...")
    # CSVs
    # 1. living/dead data vs generated (uncond)
    fn1 = f"{output_prefix}_ld_data_vs_gen.csv"
    with open(fn1,'w',newline='') as f:
        w=csv.writer(f); w.writerow(['scenario','live','dead','live_pct','dead_pct'])
        for sc,cnt,pct in [('data',d_cnt,d_pct),('gen_uncond',uc_cnt,uc_pct)]:
            w.writerow([sc,cnt['live'],cnt['dead'],pct['live'],pct['dead']])
    # plot Figure 1
    fig,ax=plt.subplots();
    for sc,pct,col in [('data',d_pct,'gray'),('gen_uncond',uc_pct,'blue')]:
        ax.bar([0,1] if sc=='data' else [0.4,1.4], [pct['live'],pct['dead']], width=0.4, label=sc,color=col)
    ax.set_xticks([0.2,1.2]); ax.set_xticklabels(['Live','Dead']); ax.legend();
    ax.set_ylabel('Percent'); fig.savefig(f"{output_prefix}_fig1.png")
    plt.close(fig)
    # 2. types data vs generated (uncond)
    fn2 = f"{output_prefix}_types_data_vs_gen.csv"
    with open(fn2,'w',newline='') as f:
        w=csv.writer(f); w.writerow(['scenario']+cats_types)
        row_data = ['data']+ [d_pct_t[c] for c in cats_types]
        row_gen = ['gen_uncond']+ [uc_pct_t[c] for c in cats_types]
        w.writerow(row_data); w.writerow(row_gen)
    fig,ax=plt.subplots(figsize=(6,4));
    x=np.arange(len(cats_types)); width=0.35
    ax.bar(x-width/2,[d_pct_t[c] for c in cats_types],width,label='data')
    ax.bar(x+width/2,[uc_pct_t[c] for c in cats_types],width,label='gen_uncond')
    ax.set_xticks(x);ax.set_xticklabels(cats_types,rotation=45);
    ax.legend();fig.tight_layout(); fig.savefig(f"{output_prefix}_fig2.png"); plt.close(fig)
    # 3. living/dead uncond vs cond
    fn3 = f"{output_prefix}_ld_uncond_vs_cond.csv"
    with open(fn3,'w',newline='') as f:
        w=csv.writer(f); w.writerow(['scenario','live','dead','pct_live','pct_dead'])
        w.writerow(['uncond',uc_cnt['live'],uc_cnt['dead'],uc_pct['live'],uc_pct['dead']])
        w.writerow(['cond_live',cl_cnt['live'],cl_cnt['dead'],cl_pct['live'],cl_pct['dead']])
        w.writerow(['cond_dead',cd_cnt['live'],cd_cnt['dead'],cd_pct['live'],cd_pct['dead']])
    fig,ax=plt.subplots();
    for i,(sc,pct,col) in enumerate([('uncond',uc_pct,'gray'),('cond_live',cl_pct,'green'),('cond_dead',cd_pct,'red')]):
        ax.bar([i-0.3,i+0.3],[pct['live'],pct['dead']],width=0.6,label=sc,color=col)
    ax.set_xticks([0,1]); ax.set_xticklabels(['Live','Dead']); ax.legend(); fig.savefig(f"{output_prefix}_fig3.png"); plt.close(fig)
    # 4 and 5 similarly for types
    fn4 = f"{output_prefix}_types_uncond_vs_cond.csv"
    with open(fn4,'w',newline='') as f:
        w=csv.writer(f); w.writerow(['scenario']+cats_types)
        w.writerow(['uncond']+ [uc_pct_t[c] for c in cats_types])
        w.writerow(['cond_live']+ [cl_pct[c] if c in cl_pct else 0 for c in cats_types])
        w.writerow(['cond_dead']+ [cd_pct[c] if c in cd_pct else 0 for c in cats_types])
    fig,ax=plt.subplots(figsize=(6,4));
    x=np.arange(len(cats_types)); wdt=0.25
    for i,sc in enumerate(['uncond','cond_live','cond_dead']):
        vals = {'uncond':uc_pct_t,'cond_live':{c:cl_pct['live'] if c!='died_out' else cl_pct['dead'] for c in cats_types},
                'cond_dead':{c:cd_pct['live'] if c!='died_out' else cd_pct['dead'] for c in cats_types}}[sc]
        ax.bar(x+i*wdt, [vals[c] for c in cats_types],wdt,label=sc)
    ax.set_xticks(x+wdt); ax.set_xticklabels(cats_types,rotation=45); ax.legend(); fig.tight_layout(); fig.savefig(f"{output_prefix}_fig4.png"); plt.close(fig)
    print(f"[generate_report] Completed evaluation for '{output_prefix}'")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--random_model',default='finished_models/model_final_random.pt')
    parser.add_argument('--quota_model',default='finished_models/model_final_quota.pt')
    parser.add_argument('--random_data',default='phase_2_conditional_diffusion/random_data_32x32')
    parser.add_argument('--quota_data',default='phase_2_conditional_diffusion/natural_data_32x32')
    parser.add_argument('--num_samples',type=int,default=1000)
    parser.add_argument('--timesteps',type=int,default=200)
    parser.add_argument('--schedule',default='cosine')
    parser.add_argument('--guidance_scale',type=float,default=1.0)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    # run for random
    evaluate(args.random_data,args.random_model,args.timesteps,args.num_samples,'random')
    # run for quota
    evaluate(args.quota_data,args.quota_model,args.timesteps,args.num_samples,'quota')
