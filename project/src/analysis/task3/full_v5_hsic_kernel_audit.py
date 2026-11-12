"""任务3 第17步：复用完整V5，检查HSIC核尺度、配对辨别力与表示梯度。

只读诊断，不更新参数、不调整基线、不利用目标地区表现选择核尺度。
复用第16步历史文件核验与来源批次；不导出逐人表示或相似度矩阵。
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import full_v5_loss_gradient_audit as previous

SETTINGS=('fixed_1','fixed_2','fixed_4','median_half','median','median_double')
PAIRS=('original','self_zc','self_zs')


def center(kernel):
    if isinstance(kernel,np.ndarray):
        return kernel-kernel.mean(0,keepdims=True)-kernel.mean(1,keepdims=True)+kernel.mean()
    return kernel-kernel.mean(0,keepdim=True)-kernel.mean(1,keepdim=True)+kernel.mean()


def hsic(k,l):
    # 任务3 第17步：对称中心化核的逐元素内积，等价于原实现的trace(Kc @ Lc)。
    return (center(k)*center(l)).sum()/(len(k)-1)**2


def rbf(z,sigma):
    return torch.exp(-torch.cdist(z,z).square()/(2*sigma*sigma))


def median_distance(z):
    value=float(torch.pdist(z.detach()).median())
    # 全部重合时保留诊断而非伪造可用的自适应尺度；回退信息写入统计表。
    return (value if value>1e-12 else 1.0),value<=1e-12


def permutations(env,count,seed):
    rng=np.random.default_rng(seed);n=len(env);out={kind:[] for kind in ('global','within_environment')}
    for _ in range(count):
        out['global'].append(rng.permutation(n))
        ids=np.arange(n)
        for group in np.unique(env):
            positions=np.flatnonzero(env==group);ids[positions]=rng.permutation(positions)
        out['within_environment'].append(ids)
    return out


def kernel_stats(k,context,block,sigma,fallback):
    a=k.detach().double().numpy();off=a[~np.eye(len(a),dtype=bool)]
    return dict(**context,block=block,sigma=sigma,median_scale_fallback=fallback,
        off_diagonal_mean=float(off.mean()),off_diagonal_median=float(np.median(off)),
        off_diagonal_p95=float(np.quantile(off,.95)),fraction_below_1e_6=float((off<1e-6).mean()),
        relative_distance_to_identity=float(np.linalg.norm(a-np.eye(len(a)))/np.sqrt(len(a))))


def pairing_stats(k,l,perms,context,pair):
    # 任务3 第17步：置换沿用同一核，不重新估计尺度；保留来源环境的置换用于识别地区混杂。
    a=center(k.detach().double().numpy());b=center(l.detach().double().numpy());den=(len(a)-1)**2
    observed=float((a*b).sum()/den);rows=[]
    for kind,indices in perms.items():
        null=np.array([(a*b[np.ix_(ids,ids)]).sum()/den for ids in indices])
        mean=float(null.mean());sd=float(null.std(ddof=1));tol=1e-12*max(1.,abs(observed))
        rows.append(dict(**context,pair=pair,shuffle=kind,hsic=observed,
            identity_reference=1/(len(a)-1),ratio_to_identity=observed*(len(a)-1),
            null_mean=mean,null_std=sd,null_q025=float(np.quantile(null,.025)),null_q975=float(np.quantile(null,.975)),
            excess_over_null=observed-mean,relative_excess=(observed-mean)/mean if abs(mean)>1e-15 else None,
            standardized_excess=(observed-mean)/sd if sd>tol else None,
            permutation_p=(1+int((null>=observed-tol).sum()))/(len(null)+1),
            numerically_flat_null=sd<=tol,permutations=len(null)))
    return rows


def inspect_batch(model,x,y,env,cfg,context,count,seed):
    logits,_,zc,zs,_=model(x)
    classification=(torch.nn.functional.binary_cross_entropy_with_logits(logits,y,pos_weight=torch.tensor([cfg.pos_weight]))+
        torch.nn.functional.binary_cross_entropy_with_logits(model.predictor_zc(zc),y,pos_weight=torch.tensor([cfg.pos_weight])))
    reference=torch.autograd.grad(classification,(zc,zs),retain_graph=True)
    medc,fallc=median_distance(zc);meds,falls=median_distance(zs)
    perm=permutations(env.numpy(),count,seed);kr=[];pr=[];gr=[]
    if cfg.hsic_sigma!=1:raise ValueError('历史核尺度不再是1，需重新核对第16步假设')
    for setting in SETTINGS:
        if setting.startswith('fixed_'):sc=ss=float(setting.split('_')[1]);fc=fs=False
        else:
            factor={'median_half':.5,'median':1.,'median_double':2.}[setting]
            sc,ss=medc*factor,meds*factor;fc,fs=fallc,falls
        ctx=dict(**context,setting=setting)
        k,l=rbf(zc,sc),rbf(zs,ss)
        kr.extend([kernel_stats(k,ctx,'zc',sc,fc),kernel_stats(l,ctx,'zs',ss,fs)])
        for pair,a,b in [('original',k,l),('self_zc',k,k),('self_zs',l,l)]:pr.extend(pairing_stats(a,b,perm,ctx,pair))
        value=cfg.lambda_independence*hsic(k,l)
        if setting=='fixed_1':
            historical=cfg.lambda_independence*previous.losses.hsic_loss(zc,zs,sigma=cfg.hsic_sigma)
            if not torch.allclose(value,historical,rtol=1e-4,atol=1e-8):raise RuntimeError('HSIC计算与历史公式不一致')
        gradient=torch.autograd.grad(value,(zc,zs),retain_graph=True)
        for block,g,c in zip(('zc','zs'),gradient,reference):
            gn=float(g.norm());cn=float(c.norm())
            if not np.isfinite(gn):raise FloatingPointError('HSIC表示梯度非有限')
            gr.append(dict(**ctx,block=block,weighted_hsic=float(value.detach()),gradient_norm=gn,
                classification_gradient_norm=cn,relative_norm=gn/cn if cn>1e-12 else None))
    return kr,pr,gr


def summarize(kernels,pairings,gradients):
    k=kernels.groupby(['held_out','mode','setting','block'],as_index=False).agg(
        mean_sigma=('sigma','mean'),mean_off_diagonal=('off_diagonal_mean','mean'),
        mean_fraction_below_1e_6=('fraction_below_1e_6','mean'),
        mean_relative_distance_to_identity=('relative_distance_to_identity','mean'),
        fallback_batches=('median_scale_fallback','sum'),batches=('batch','count'))
    pairings=pairings.assign(above=pairings.hsic>pairings.null_q975)
    p=pairings.groupby(['held_out','mode','setting','pair','shuffle'],as_index=False).agg(
        mean_hsic=('hsic','mean'),mean_ratio_to_identity=('ratio_to_identity','mean'),
        mean_null=('null_mean','mean'),mean_excess=('excess_over_null','mean'),
        mean_relative_excess=('relative_excess','mean'),mean_standardized_excess=('standardized_excess','mean'),
        flat_null_batches=('numerically_flat_null','sum'),
        batches_above_null_q975=('above','sum'),batches=('batch','count'))
    # 不合并不同批次的P值，不把重叠批次当成独立人群。
    g=gradients.groupby(['held_out','mode','setting','block'],as_index=False).agg(
        mean_weighted_hsic=('weighted_hsic','mean'),mean_gradient_norm=('gradient_norm','mean'),
        mean_relative_norm=('relative_norm','mean'),batches=('batch','count'))
    return k,p,g


def draw(k,p,g,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    settings=list(SETTINGS);x=np.arange(len(settings))
    fig,axes=plt.subplots(1,3,figsize=(15,4),sharey=True)
    for ax,region in zip(axes,previous.REGIONS):
        part=k[(k.held_out==region)&(k['mode']=='train_dropout')]
        for block in ('zc','zs'):
            vals=part[part.block==block].set_index('setting').reindex(settings).mean_fraction_below_1e_6
            ax.plot(x,vals,'o-',label=block)
        ax.set_title(region);ax.set_xticks(x,settings,rotation=35,ha='right');ax.set_ylim(-.03,1.03);ax.legend()
    axes[0].set_ylabel('Fraction of off-diagonal similarities < 1e-6')
    fig.suptitle('Source kernels: near-zero similarity between different people')
    fig.tight_layout();fig.savefig(out/'figures/01_kernel_off_diagonal_saturation.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4),sharey=False)
    for ax,region in zip(axes,previous.REGIONS):
        part=p[(p.held_out==region)&(p['mode']=='train_dropout')&(p.pair=='original')]
        for kind in ('global','within_environment'):
            vals=part[part.shuffle==kind].set_index('setting').reindex(settings).mean_relative_excess
            ax.plot(x,vals,'o-',label=kind)
        ax.axhline(0,color='grey',lw=1);ax.set_title(region);ax.set_xticks(x,settings,rotation=35,ha='right');ax.legend(fontsize=8)
    axes[0].set_ylabel('Relative HSIC excess over shuffle')
    fig.suptitle('Source pairing sensitivity; bandwidth changes are diagnostic only')
    fig.tight_layout();fig.savefig(out/'figures/02_pairing_vs_shuffle_controls.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4),sharey=True)
    for ax,region in zip(axes,previous.REGIONS):
        part=p[(p.held_out==region)&(p['mode']=='train_dropout')&(p.shuffle=='global')]
        for pair in PAIRS:
            vals=part[part.pair==pair].set_index('setting').reindex(settings).mean_relative_excess
            ax.plot(x,vals,'o-',label=pair)
        ax.axhline(0,color='grey',lw=1);ax.set_title(region);ax.set_xticks(x,settings,rotation=35,ha='right');ax.legend(fontsize=8)
    axes[0].set_ylabel('Relative HSIC excess over global shuffle')
    fig.suptitle('Self-pair positive controls: can the kernel detect known dependence?')
    fig.tight_layout();fig.savefig(out/'figures/03_self_pair_positive_controls.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4),sharey=True)
    for ax,region in zip(axes,previous.REGIONS):
        part=g[g.held_out==region]
        for mode in previous.MODES:
            for block in ('zc','zs'):
                vals=part[(part['mode']==mode)&(part.block==block)].set_index('setting').reindex(settings).mean_relative_norm
                vals=vals.where(vals>0)
                ax.plot(x,vals,marker='o',ls='-' if mode=='train_dropout' else '--',label=f'{mode}: {block}')
        ax.set_yscale('log');ax.set_title(region);ax.set_xticks(x,settings,rotation=35,ha='right');ax.legend(fontsize=7)
    axes[0].set_ylabel('HSIC / classification gradient norm')
    fig.suptitle('Fixed representations; detached median bandwidth; no optimizer updates')
    fig.tight_layout();fig.savefig(out/'figures/04_hsic_representation_gradient_sensitivity.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous',type=Path,default=previous.ROOT/'outputs/v2-region-loro-selection91-42')
    parser.add_argument('--output',type=Path,default=previous.ROOT/'outputs/analysis_2611/task3_step17_full_v5_hsic_kernel_audit')
    parser.add_argument('--batches',type=int,default=12)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--permutations',type=int,default=99)
    args=parser.parse_args()
    if args.batches<2 or args.batch_size<4 or args.permutations<19:raise ValueError('至少2批次、批次大小4、置换19次')
    for region in previous.REGIONS:
        for name in ('config.json','training.json','best_model.pt','splits.npz'):previous.require_file(args.previous/region/'v5'/name)
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'figures').mkdir()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    meta=dict(completed=False,training_runs=0,optimizer_steps=0,model_loads=0,source_data_only=True,
        individual_records_exported=False,batches=args.batches,batch_size=args.batch_size,permutations=args.permutations,
        settings=SETTINGS,runner_sha256=previous.digest(Path(__file__)),
        helper_sha256=previous.digest(Path(previous.__file__)),
        code_sha256={Path(module.__file__).name:previous.digest(Path(module.__file__))
            for module in (previous.models,previous.losses)},
        torch_version=str(torch.__version__),numpy_version=np.__version__,
        limitations=['仅检查最终完整V5，不还原历史训练轨迹；核退化不等于原始数据或表示崩溃。',
          '核尺度预先固定；中位数按当前来源批次、各块分别估计并停止求导，只用于诊断，不选择新训练配置。',
          '原始HSIC值跨尺度不可直接比较；结合置换超额、自配对正对照与梯度解读。',
          '全局置换破坏所有对应；环境内置换保留来源地区混杂。环境内无超额不证明因果独立。',
          '同一来源总体重复抽样，不合并置换P值，不代表跨种子或独立队列重复。',
          '表示坐标上的梯度比值不是第16步参数组比值，也不是Adam实际更新量。',
          '即使确认HSIC缺乏辨别力，也尚未证明它造成东部泛化差。'])
    previous.dump(args.output/'experiment_metadata.json',meta)
    all_k=[];all_p=[];all_g=[];checks=[]
    for index,region in enumerate(previous.REGIONS):
        with torch.random.fork_rng(devices=[]):
            model,cfg,(x,y,env),info=previous.load_fold(args.previous/region/'v5',region)
            before=previous.state_hash(model);batches=previous.source_batches(env,args.batch_size,args.batches,26111600+index)
            h=hashlib.sha256()
            for ids in batches:h.update(ids.astype(np.int64).tobytes())
            for mode in previous.MODES:
                model.train(mode=='train_dropout')
                for repeat,ids in enumerate(batches):
                    torch.manual_seed(261116000+index*1000+repeat)
                    values=inspect_batch(model,torch.as_tensor(x[ids]),torch.as_tensor(y[ids]),torch.as_tensor(env[ids]),cfg,
                        dict(held_out=region,mode=mode,batch=repeat),args.permutations,261117000+index*1000+repeat)
                    for collection,rows in zip((all_k,all_p,all_g),values):collection.extend(rows)
            unchanged=before==previous.state_hash(model);untouched=all(p.grad is None for p in model.parameters())
            if not unchanged or not untouched:raise RuntimeError('只读诊断改变了模型或优化梯度')
            checks.append(dict(held_out=region,model_state_unchanged=unchanged,optimizer_gradients_untouched=untouched,
                selected_epoch=info['selected_epoch'],common_mode_batch_sha256=h.hexdigest(),state_sha256=before))
            meta.setdefault('fold_artifacts',{})[region]=info
        meta['model_loads']+=1
        print(f'任务3 第17步：{region}核尺度与配对对照已完成，参数未更新',flush=True)
    k,p,g=map(pd.DataFrame,(all_k,all_p,all_g));ks,ps,gs=summarize(k,p,g)
    for name,frame in dict(kernel_batches=k,pairing_batches=p,gradient_batches=g,kernel_summary=ks,
        pairing_summary=ps,gradient_summary=gs,control_checks=pd.DataFrame(checks)).items():
        frame.to_csv(args.output/(name+'.csv'),index=False,encoding='utf-8-sig')
    draw(ks,ps,gs,args.output)
    meta['completed']=True;previous.dump(args.output/'experiment_metadata.json',meta)
    print(f'任务3 第17步完成：训练0次，优化0步。输出目录：{args.output}',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
